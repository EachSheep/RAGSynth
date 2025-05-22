import os
import glob
import json
import time
import random
import yaml
import numpy as np
import logging
import pytz
import copy
import torch
from torch import Tensor
import torch.distributed as dist
from torch.optim import AdamW
from torch.utils.data import DataLoader, Sampler, Dataset
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator

from peft import LoraConfig, TaskType

import faiss
# from faiss import IndexFlatIP, StandardGpuResources, GpuIndexFlatIP, GpuIndexFlatConfig
import argparse
from tqdm import tqdm
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import batch_to_device, truncate_embeddings
from transformers import get_cosine_schedule_with_warmup
from collections import defaultdict

from rag.embed.embed_query import process_questions
from rag.embed.embed_tune_model_utils import get_query_positive_map_from_contents, calculate_all_embeddings

from rag.utils.metric_calculator import (
    eval_retrieval_results_for_file_content,
    eval_retrieval_results_for_file_entity_graph
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CUSTOM_CORPUS_HOME = os.getenv("CUSTOM_CORPUS_HOME", None)
if CUSTOM_CORPUS_HOME is None:
    raise ValueError("CUSTOM_CORPUS_HOME environment variable is not set")

def load_base_model(embed_model_path, torch_dtype=torch.float32, max_seq_length=32768):
    """Load the base model without LoRA configuration."""
    
    logging.getLogger().setLevel(logging.WARNING)

    model = SentenceTransformer(
        embed_model_path,
        trust_remote_code=True,
        model_kwargs={"torch_dtype": torch_dtype}
    )
    
    model.max_seq_length = max_seq_length

    logging.getLogger().setLevel(logging.INFO)

    return model

def add_lora_to_model(model, target_modules, exclude_modules, lora_rank=8, lora_alpha=32, lora_dropout=0.1):
    """Configure the given model with LoRA support."""
    
    # Define LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,  # Adjust based on your task
        inference_mode=False,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,  # Adjust based on your model's architecture
        exclude_modules=exclude_modules,  # Modules to exclude from LoRA
    )
    
    model.add_adapter(lora_config)

    return model

def load_content(query_file_path):
    """
    Load contents from a JSON file.
    """
    with open(query_file_path, "r") as f:
        contents = json.load(f)
    return contents


class RetrievalDataset(Dataset):
    def __init__(self, corpuspath_2_query2positive, corpuspath_2_docid2context):
        """
        Args:
            corpuspath_2_query2positive (dict): 
                Dictionary mapping corpus paths to query-to-positiveids mappings.
            corpuspath_2_docid2context (dict): 
                Dictionary mapping doc IDs to their corresponding context.
        """
        self.corpuspath_2_query2positive = corpuspath_2_query2positive
        self.corpuspath_2_docid2context = corpuspath_2_docid2context

        # Initialize data structures
        self.corpuspaths = list(self.corpuspath_2_query2positive.keys())
        self.corpuspath_2_indices = {}  # Mapping from corpus path to list of sample indices
        self.data = []  # List to hold all data samples

        # Populate self.data and self.corpuspath_2_indices
        for corpuspath in self.corpuspaths:
            query2positive = self.corpuspath_2_query2positive[corpuspath]
            for query, postive_dict in query2positive.items():
                positive_contexts = []
                failed = False
                for pid in postive_dict["positive_indices"]:
                    if pid in self.corpuspath_2_docid2context[corpuspath]:
                        positive_contexts.append(self.corpuspath_2_docid2context[corpuspath][pid])
                    else:
                        failed = True
                        break
                if failed or not positive_contexts:
                    continue
                item = {
                    'query': query,
                    'positiveids': positive_contexts,
                    'corpuspath': corpuspath  # Keep track of the corpus
                }
                self.data.append(item)
                idx = len(self.data) - 1
                self.corpuspath_2_indices.setdefault(corpuspath, []).append(idx)

    def __len__(self):
        raise NotImplementedError("Use CorpusBatchSampler to determine the number of batches.")

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'query': item['query'],
            'positiveids': item['positiveids'],
            'corpuspath': item['corpuspath']
        }

class StepWiseCorpusBatchSampler(Sampler):
    def __init__(self, corpuspath_2_indices, batch_size, drop_last=False, seed=42):
        """
        Args:
            corpuspath_2_indices (dict): Mapping from corpus paths to list of sample indices.
            batch_size (int): Number of samples per batch per process.
            drop_last (bool): Whether to drop the last batch if it's smaller than batch_size.
            seed (int): Base random seed for shuffling.
        """
        self.corpuspath_2_indices = corpuspath_2_indices
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.base_seed = seed
        self.epoch = 0  # Initialize epoch counter

        # Preprocess: Create list of batches per corpuspath
        self.corpuspath_to_batches = defaultdict(list)
        for corpus, indices in self.corpuspath_2_indices.items():
            shuffled = indices.copy()
            random.Random(self.base_seed).shuffle(shuffled)
            # Split into batches
            for i in range(0, len(shuffled), self.batch_size):
                batch = shuffled[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    self.corpuspath_to_batches[corpus].append(batch)

        # Initial shuffle of corpuspaths
        self.corpuspaths = list(self.corpuspath_to_batches.keys())
        random.Random(self.base_seed).shuffle(self.corpuspaths)

        # Create the initial list of batches
        self.batches = []
        for corpus in self.corpuspaths:
            self.batches.extend([(corpus, batch) for batch in self.corpuspath_to_batches[corpus]])

        self.num_batches = len(self.batches)

        # Shuffle batches for the first epoch
        self.shuffle_batches()

    def shuffle_batches(self):
        """
        Shuffle the corpuspaths and batches using a seed that varies with each epoch.
        This ensures a different shuffle order for each epoch.
        """
        # Create a new random instance with a seed that changes every epoch
        random_seed = self.base_seed + self.epoch
        rand_instance = random.Random(random_seed)

        # Shuffle corpuspaths
        rand_instance.shuffle(self.corpuspaths)

        # Recreate the batches list based on the shuffled corpuspaths
        self.batches = []
        for corpus in self.corpuspaths:
            self.batches.extend([(corpus, batch) for batch in self.corpuspath_to_batches[corpus]])

        # Shuffle the entire batches list to mix batches from different corpuspaths
        rand_instance.shuffle(self.batches)

    def __iter__(self):
        """
        Yield batches assigned to the current process in a distributed setting.
        """
        for batch_group in self._group_batches():
            # Assign each process its respective batch
            batch = batch_group['batches'][dist.get_rank()]
            yield batch

    def _group_batches(self):
        """
        Groups batches into step-wise groups where each group contains world_size batches 
        from the same corpuspath. This ensures that each process works on different 
        corpuspaths in a synchronized manner.
        """
        world_size = dist.get_world_size()
        grouped_batches = []

        # Group batches by corpuspath
        corpuspath_to_batches = defaultdict(list)
        for corpuspath, batch in self.batches:
            corpuspath_to_batches[corpuspath].append(batch)

        # Create step-wise groups
        for corpuspath, batches in corpuspath_to_batches.items():
            # Split batches into chunks of world_size
            for i in range(0, len(batches), world_size):
                batch_chunk = batches[i:i + world_size]
                if len(batch_chunk) == world_size or not self.drop_last:
                    grouped_batches.append({
                        'corpuspath': corpuspath,
                        'batches': batch_chunk
                    })

        # Shuffle the grouped batches to ensure different order each epoch
        rand_instance = random.Random(self.base_seed + self.epoch)
        rand_instance.shuffle(grouped_batches)

        return grouped_batches

    def __len__(self):
        """
        Returns the total number of step-wise grouped batches.
        This is used by the DataLoader to determine how many iterations are in an epoch.
        """
        world_size = dist.get_world_size()
        return len(self.batches) // world_size

    def reset(self):
        """
        Resets the sampler for a new epoch.
        Increments the epoch counter and reshuffles the batches.
        """
        self.epoch += 1
        self.shuffle_batches()

def create_train_dataloader(dataset, corpuspath_2_indices, train_micro_batch_size_per_gpu, drop_last, seed):
    batch_sampler = StepWiseCorpusBatchSampler(
        corpuspath_2_indices=corpuspath_2_indices,
        batch_size=train_micro_batch_size_per_gpu,
        drop_last=drop_last,
        seed=seed
    )
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=custom_collate_fn
    )

def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-length lists in the batch.
    Pads the positive lists and creates a mask.
    """
    queries = [item['query'] for item in batch]
    positives = [item['positiveids'] for item in batch]
    corpuspaths = [item['corpuspath'] for item in batch]

    return {
        'query': queries,
        'positives': positives,
        'corpuspaths': corpuspaths
    }


def save_checkpoint(state, checkpoint_dir, global_step, epoch, max_checkpoints=20, rank=0, is_best=False):
    """
    Save the training checkpoint.
    Only the master process (rank 0) should save checkpoints.
    If is_best is True, save a separate best checkpoint.
    """
    if rank != 0:
        return

    checkpoint_dir = os.path.join(CUSTOM_CORPUS_HOME, checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)

    if is_best:
        checkpoint_path = os.path.join(checkpoint_dir, f"best_checkpoint_global_step_{global_step}_epoch_{epoch}.pt")
        torch.save(state, checkpoint_path)
        logger.info(f"Best checkpoint saved at {checkpoint_path}")

        # Remove old best checkpoints
        best_checkpoints = glob.glob(os.path.join(checkpoint_dir, 'best_checkpoint_global_step_*.pt'))
        for cp in best_checkpoints:
            if cp != checkpoint_path:
                os.remove(cp)
                logger.info(f"Removed old best checkpoint {cp}")
    else:
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_global_step_{global_step}_epoch_{epoch}.pt")
        torch.save(state, checkpoint_path)
        logger.info(f"Checkpoint saved at {checkpoint_path}")

        # Manage the number of regular checkpoints
        checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, 'checkpoint_global_step_*_epoch_*.pt')), key=os.path.getctime)
        if len(checkpoints) > max_checkpoints:
            to_remove = checkpoints[:-max_checkpoints]
            for cp in to_remove:
                os.remove(cp)
                logger.info(f"Removed old checkpoint {cp}")

def get_latest_checkpoint_path(checkpoint_dir, rank=0):
    """
    Get the latest checkpoint path.
    Only the master process (rank 0) should perform this.
    """
    if rank != 0:
        return None

    checkpoints = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_global_step_*_epoch_*.pt'))
    if not checkpoints:
        return None
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    return latest_checkpoint


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_arguments(rank):

    parser = argparse.ArgumentParser(description="Fine-tune the model and find similar documents for queries.")

    parser.add_argument("--ds_config_file", type=str, required=True, help="Path to the DeepSpeed configuration file.")
    parser.add_argument("--train_micro_batch_size_per_gpu", type=int, help="Batch size per GPU for training.") # load from ds_config_file
    parser.add_argument("--gradient_accumulation_steps", type=int, help="Gradient accumulation steps.")

    parser.add_argument("--query_prefix", type=str, default=None, help="Query prefix.")
    parser.add_argument("--prompt_name", type=str, default=None, help="Prompt name.")
    parser.add_argument("--max_seq_length", type=int, default=None, help="Maximum sequence length.")
    parser.add_argument("--add_eos", action="store_true", help="Add EOS token to each input example.")
    parser.add_argument("--normalize_embeddings", action="store_true", help="Normalize embeddings before calculating similarity.")
    
    parser.add_argument('--corpuspath_2_inputpath', type=str, help="Path to the corpus 2 input map JSON file.")
    parser.add_argument('--eval_corpuspath_2_inputpath', type=str, help="Path to the corpus 2 input map JSON file.")
    parser.add_argument("--embed_model_path", type=str, help="Path to the pre-trained model directory.")
    parser.add_argument("--embed_model_name", type=str, help="Name of the pre-trained model.")
    parser.add_argument("--output_model_dir", type=str, default=None, help="Path to save the fine-tuned model.")
    parser.add_argument("--not_use_rephrased", default=False, action='store_true', help="Use Rephrased data to train or not")
    parser.add_argument("--not_use_rephrased_part", default=False, action='store_true', help="Use Rephrased part data to train or not")
    parser.add_argument("--not_use_rephrased_hybrid", default=False, action='store_true', help="Use Rephrased hybrid data to train or not")

    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for fine-tuning.")
    parser.add_argument("--loss_margin", type=float, default=0.2, help="Loss margin for ANCE loss.")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs for fine-tuning.")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Maximum number of training steps.")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Number of warmup steps for fine-tuning.")
    parser.add_argument("--num_negatives", type=int, default=3, help="Number of negative samples for each positive sample.")
    parser.add_argument("--max_train_num", type=int, default=None, help="Maximum number of training examples.")
    parser.add_argument("--max_norm", type=int, default=1, help="Maximum norm for gradient clipping.")
    parser.add_argument("--lambda_reg", type=float, default=0.9, help="Regularization weight to prevent overfitting.")
    parser.add_argument("--lora_rank", type=int, default=8, help="Rank for LoRA.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="Alpha for LoRA.")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="Dropout for LoRA.")
    parser.add_argument("--max_checkpoints", type=int, default=20, help="Maximum number of checkpoints to keep.")

    parser.add_argument("--do_eval", action='store_true', help="Evaluate the model.")
    parser.add_argument("--eval_steps", type=int, default=1, help="Frequency (in epochs) to evaluate the model.")
    parser.add_argument("--max_eval_num", type=int, default=-1, help="Maximum number of eval chunks.")
    parser.add_argument("--query_batch_size", type=int, default=8, help="Batch size for encoding contexts.")
    parser.add_argument('--top_k_values', type=int, nargs='+', default=[3, 5], help="List of top_k values for which precision will be calculated.")
    parser.add_argument("--top_k", type=int, default=20, help="Number of top documents to retrieve.")
    parser.add_argument('--only_eval_at_rephrased_poses', type=int, nargs='+', default=[3], help="List of rephrased positions to evaluate.")
    parser.add_argument('--only_eval_at_rephrased_poses_part', type=int, nargs='+', default=[2], help="List of rephrased positions part to evaluate.")
    parser.add_argument('--only_eval_at_rephrased_poses_hybrid', type=int, nargs='+', default=[6], help="List of rephrased positions hybrid to evaluate.")
    parser.add_argument('--difference_alpha', type=float, default=1.0, help="The difference_alpha parameter for the softmax function in nDCG calculation.")

    parser.add_argument("--resume_from_checkpoint", action='store_true', help="Resume training from a checkpoint.")
    parser.add_argument("--checkpoint_freq", type=int, help="Frequency (in epochs) to save checkpoints.")
    parser.add_argument("--log_dir", type=str, help="Directory for TensorBoard logs.")

    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    parser.add_argument(
        "--faiss_update_freq",
        type=int,
        default=1,
        help="Frequency to update FAISS index. If using epoch-based frequency, set to N to update every N epochs."
    )

    args = parser.parse_args()

    # Load DeepSpeed config
    with open(args.ds_config_file, "r") as f:
        ds_config = json.load(f)
    args.train_micro_batch_size_per_gpu = ds_config["train_micro_batch_size_per_gpu"]
    args.gradient_accumulation_steps = ds_config["gradient_accumulation_steps"]

    # Process corpus paths
    args.corpuspath_2_inputpath = json.loads(args.corpuspath_2_inputpath.replace("'", "\""))
    new_chunk_input_map = {}
    for chunk_path_key in args.corpuspath_2_inputpath:
        new_chunk_input_map[os.path.join(CUSTOM_CORPUS_HOME, chunk_path_key)] = [
            os.path.join(CUSTOM_CORPUS_HOME, inputpath) for inputpath in args.corpuspath_2_inputpath[chunk_path_key]
        ]
    args.corpuspath_2_inputpath = new_chunk_input_map

    args.eval_corpuspath_2_inputpath = json.loads(args.eval_corpuspath_2_inputpath.replace("'", "\""))
    new_chunk_input_map = {}
    for chunk_path_key in args.eval_corpuspath_2_inputpath:
        new_chunk_input_map[os.path.join(CUSTOM_CORPUS_HOME, chunk_path_key)] = [
            os.path.join(CUSTOM_CORPUS_HOME, inputpath) for inputpath in args.eval_corpuspath_2_inputpath[chunk_path_key]
        ]
    args.eval_corpuspath_2_inputpath = new_chunk_input_map

    # Define log and model directories
    args.log_dir = os.path.join(CUSTOM_CORPUS_HOME, args.log_dir)
    args.output_model_dir = os.path.join(CUSTOM_CORPUS_HOME, args.output_model_dir)

    # Initialize time_str only once to ensure consistency across processes
    if rank == 0:
        # time_str = datetime.now().strftime('%Y_%b%d_%H-%M-%S')
        beijing_tz = pytz.timezone('Asia/Shanghai')
        time_str = datetime.now(beijing_tz).strftime('%Y_%b%d_%H-%M-%S')
        # Save the time_str to a file for other ranks to read
        time_str_path = os.path.join(args.output_model_dir, "time_str.txt")
        os.makedirs(args.output_model_dir, exist_ok=True)
        with open(time_str_path, "w") as f:
            f.write(time_str)
    else:
        time_str_path = os.path.join(args.output_model_dir, "time_str.txt")
    
    # Ensure rank 0 has written the time_str before others read it
    if rank != 0:
        while not os.path.exists(time_str_path):
            time.sleep(1)  # Wait for rank 0 to write the file

    # Read the shared time_str
    with open(time_str_path, "r") as f:
        time_str = f.read().strip()

    args.output_model_dir = os.path.join(args.output_model_dir, time_str)

    # Define checkpoint and log directories using the shared time_str
    args.checkpoint_dir = os.path.join(args.output_model_dir, "checkpoints")
    args.log_dir = os.path.join(args.log_dir, time_str, "logs")

    # wait for all ranks to execute to here
    dist.barrier()
    if rank == 0:
        os.remove(time_str_path)

    # Create directories only once to avoid race conditions
    if rank == 0:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        # Save the arguments to args.json
        with open(os.path.join(args.output_model_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
    else:
        # Wait until directories are created by rank 0
        while not os.path.exists(args.checkpoint_dir) or not os.path.exists(args.log_dir):
            time.sleep(1)

    return args


class ANCELoss(torch.nn.Module):

    def __init__(self, model, faiss_index, id_2_docid, query_2_positive, docid_2_context, device, num_negatives=3, loss_margin=0.2, normalize_embeddings=True):
        super(ANCELoss, self).__init__()
        self.model = model
        self.index = faiss_index
        self.id_2_docid = id_2_docid  # Maps Faiss index to document ID
        self.query_2_positive = query_2_positive # Maps query to positive doc IDs
        self.docid_2_context = docid_2_context  # Maps document ID to document text
        self.device = device
        self.num_negatives = num_negatives
        self.loss_margin = loss_margin
        self.normalize_embeddings = normalize_embeddings
        self.loss_fn = torch.nn.MarginRankingLoss(margin=loss_margin)
    
    def tmp_encode(
        self,
        model,
        sentences: list[str],
        prompt_name: str | None = None,
        prompt: str | None = None,
        device: str = None,
        normalize_embeddings: bool = False,
        **kwargs,
    ) -> list[Tensor] | np.ndarray | Tensor:
        """
        Computes sentence embeddings.

        Args:
            sentences (Union[str, List[str]]): The sentences to embed.
            prompt_name (Optional[str], optional): The name of the prompt to use for encoding. Must be a key in the `prompts` dictionary,
                which is either set in the constructor or loaded from the model configuration. For example if
                ``prompt_name`` is "query" and the ``prompts`` is {"query": "query: ", ...}, then the sentence "What
                is the capital of France?" will be encoded as "query: What is the capital of France?" because the sentence
                is appended to the prompt. If ``prompt`` is also set, this argument is ignored. Defaults to None.
            prompt (Optional[str], optional): The prompt to use for encoding. For example, if the prompt is "query: ", then the
                sentence "What is the capital of France?" will be encoded as "query: What is the capital of France?"
                because the sentence is appended to the prompt. If ``prompt`` is set, ``prompt_name`` is ignored. Defaults to None.
            device (str, optional): Which :class:`torch.device` to use for the computation. Defaults to None.
            normalize_embeddings (bool, optional): Whether to normalize returned vectors to have length 1. In that case,
                the faster dot-product (util.dot_score) instead of cosine similarity can be used. Defaults to False.

        Returns:
            Union[List[Tensor], ndarray, Tensor]: By default, a 2d numpy array with shape [num_inputs, output_dimension] is returned.

        Example:
            ::

                from sentence_transformers import SentenceTransformer

                # Load a pre-trained SentenceTransformer model
                model = SentenceTransformer('all-mpnet-base-v2')

                # Encode some texts
                sentences = [
                    "The weather is lovely today.",
                    "It's so sunny outside!",
                    "He drove to the stadium.",
                ]
                embeddings = model.encode(sentences)
                print(embeddings.shape)
                # (3, 768)
        """

        if prompt is None:
            if prompt_name is not None:
                try:
                    prompt = model.prompts[prompt_name]
                except KeyError:
                    raise ValueError(
                        f"Prompt name '{prompt_name}' not found in the configured prompts dictionary with keys {list(model.prompts.keys())!r}."
                    )
            elif model.default_prompt_name is not None:
                prompt = model.prompts.get(model.default_prompt_name, None)
        else:
            if prompt_name is not None:
                logger.warning(
                    "Encode with either a `prompt`, a `prompt_name`, or neither, but not both. "
                    "Ignoring the `prompt_name` in favor of `prompt`."
                )

        extra_features = {}
        if prompt is not None:
            sentences = [prompt + sentence for sentence in sentences]

            # Some models (e.g. INSTRUCTOR, GRIT) require removing the prompt before pooling
            # Tracking the prompt length allow us to remove the prompt during pooling
            tokenized_prompt = model.tokenize([prompt])
            if "input_ids" in tokenized_prompt:
                extra_features["prompt_length"] = tokenized_prompt["input_ids"].shape[-1] - 1

        if device is None:
            device = model.device

        model.to(device)

        features = model.tokenize(sentences)
        features = batch_to_device(features, device)
        features.update(extra_features)

        out_features = model.forward(features, **kwargs)
        out_features["sentence_embedding"] = truncate_embeddings(
            out_features["sentence_embedding"], model.truncate_dim
        )

        embeddings = out_features["sentence_embedding"]
        if normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings
      
    def forward(self, queries, positive_list, masks):
        """
        Calculates the ANCE loss.

        Args:
            queries: list of query strings
            positive_list: list of lists of positive passage strings (padded)
            masks: Tensor of shape (batch_size, max_num_positives) indicating valid positives
        """
        batch_size = len(queries)

        # Encode the queries
        query_embeddings = self.tmp_encode(
            self.model,
            queries,
            normalize_embeddings=self.normalize_embeddings,
            device=self.device,
        )  # Shape: (batch_size, embedding_dim)
        
        # Encode all positive samples (including padding)
        # Flatten the list for encoding
        flattened_positives = [p for sub_positivelist in positive_list for p in sub_positivelist]
        positive_embeddings = self.tmp_encode(
            self.model,
            flattened_positives,
            normalize_embeddings=self.normalize_embeddings,
            device=self.device,
        )  # Shape: (batch_size * max_num_positives, embedding_dim)

        # Get max_num_positives from the shape of masks
        max_num_positives = masks.shape[1]

        # Add an assertion to ensure shapes match
        assert positive_embeddings.shape[0] == batch_size * max_num_positives, \
            f"Expected {batch_size * max_num_positives} positive embeddings, but got {positive_embeddings.shape[0]}"

        # Reshape to (batch_size, max_num_positives, embedding_dim)
        positive_embeddings = positive_embeddings.view(batch_size, max_num_positives, -1)

        # Instead of setting invalid embeddings to -inf, we'll use the mask to compute scores selectively
        # Calculate positive sample scores: For each query, compute the dot product with all valid positives
        # query_embeddings: (batch_size, embedding_dim)
        # positive_embeddings: (batch_size, max_num_positives, embedding_dim)
        # Calculate dot product
        # Expand query_embeddings to (batch_size, max_num_positives, embedding_dim)
        query_embeddings_expanded = query_embeddings.unsqueeze(1).expand(-1, max_num_positives, -1)
        positive_scores = torch.sum(query_embeddings_expanded * positive_embeddings, dim=2)  # Shape: (batch_size, max_num_positives)

        # Mask the positive_scores to ignore invalid positives
        # masks: (batch_size, max_num_positives)
        masked_positive_scores = positive_scores * masks  # Zero out scores where mask is 0

        # Compute the sum and count of valid positives per query for averaging
        sum_positive_scores = masked_positive_scores.sum(dim=1)  # Shape: (batch_size,)
        count_positive = masks.sum(dim=1)  # Shape: (batch_size,)
        # To avoid division by zero, ensure that count_positive is at least 1
        count_positive = count_positive.clamp(min=1)
        # Calculate the average positive score for each query
        average_positive_scores = sum_positive_scores / count_positive  # Shape: (batch_size,)

        # Convert query embeddings to a numpy array for use with Faiss
        query_embeddings_np = query_embeddings.detach().cpu().to(torch.float32).numpy()

        # Retrieve indices of negative samples using Faiss
        _, I = self.index.search(query_embeddings_np, self.num_negatives * 3)  # Retrieve more to exclude positives

        # Collect negative samples for each query
        negatives = []
        for i in range(batch_size):
            # Exclude indices of positive samples for the current query (adjust as needed based on your data structure)
            current_query = queries[i]
            current_query_positives = set(self.query_2_positive[current_query]['positive_indices'])
            negative_indices = [idx for idx in I[i] if self.id_2_docid[idx] not in current_query_positives]

            # Select num_negatives negative samples
            selected_negatives = negative_indices[:self.num_negatives]

            # If there are not enough negative samples, raise an error (you can adjust this, e.g., re-retrieve or sample randomly)
            if len(selected_negatives) < self.num_negatives:
                raise ValueError(f"Not enough negative samples for query: {current_query}")

            negatives.extend([self.docid_2_context[self.id_2_docid[idx]] for idx in selected_negatives])

        # Encode negative samples
        negative_embeddings = self.tmp_encode(
            self.model,
            negatives,
            normalize_embeddings=self.normalize_embeddings,
            device=self.device
        ).view(batch_size, self.num_negatives, -1)  # Shape: (batch_size, num_negatives, embedding_dim)

        # Calculate negative sample scores
        # query_embeddings_expanded: (batch_size, 1, embedding_dim) -> (batch_size, num_negatives, embedding_dim)
        query_embeddings_expanded_neg = query_embeddings.unsqueeze(1).expand(-1, self.num_negatives, -1)
        negative_scores = torch.sum(query_embeddings_expanded_neg * negative_embeddings, dim=2)  # Shape: (batch_size, num_negatives)

        # Define the target tensor, indicating that positive scores should be higher than negative scores
        target = torch.ones_like(negative_scores).to(self.device)
        loss = self.loss_fn(
            average_positive_scores.unsqueeze(1).expand_as(negative_scores),
            negative_scores,
            target
        )

        return loss

def main():
    accelerator = Accelerator()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    # if rank != 0:
    #     logger.setLevel(logging.WARNING)
    # else:
    #     logger.setLevel(logging.INFO)

    device = torch.device("cuda", local_rank)

    args = parse_arguments(rank)
    set_seed(args.seed)
    
    torch_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else \
                  torch.float16 if device.type == "cuda" else torch.float32
    if rank == 0:
        logging.info(f"Using {torch_dtype} precision.")
    
    base_model = load_base_model(
        embed_model_path=args.embed_model_path,
        torch_dtype=torch_dtype,
        max_seq_length=args.max_seq_length
    )
    trainable_params, exclude_modules = [], []
    if args.embed_model_name == "gte-multilingual-base":
        trainable_params = [
            "attention.qkv_proj",
            "attention.o_proj",
            "mlp.up_gate_proj",
            "mlp.down_proj",
        ]
        # for name, param in base_model.named_parameters():
        #     print(name)
        # raise NotImplementedError("Fine-tuning gte-multilingual-base is not supported.")
    elif args.embed_model_name == "snowflake-arctic-embed-m-long":
        trainable_params = [
            "attn.Wqkv",
            "attn.out_proj",
            "mlp.fc11",
            "mlp.fc12",
            "mlp.fc2",
        ]
        # for name, param in base_model.named_parameters():
        #     print(name)
        # raise NotImplementedError("Fine-tuning snowflake-arctic-embed-m-long is not supported.")
    elif args.embed_model_name == "rubert-tiny-turbo":
        trainable_params = [
            "attention.self.query",
            "attention.self.key",
            "attention.self.value",
            "attention.output.dense",
            "intermediate.dense",
            "output.dense",
        ]
        # for name, param in base_model.named_parameters():
        #     print(name)
        # raise NotImplementedError("Fine-tuning rubert-tiny-turbo is not supported.")
    elif args.embed_model_name == "stella_en_400M_v5":
        trainable_params = [
            "attention.qkv_proj",
            "attention.o_proj",
            "mlp.up_gate_proj", # mlp.up_gate_proj
            "mlp.down_proj",
            "linear",
        ]
    elif args.embed_model_name == "snowflake-arctic-embed-m-v1.5":
        trainable_params = [
            "attention.self.query",
            "attention.self.key",
            "attention.self.value",
            "attention.output.dense",
            "intermediate.dense",
            "output.dense",
        ]
        # for name, param in base_model.named_parameters():
        #     print(name)
        # raise NotImplementedError("Fine-tuning snowflake-arctic-embed-m-v1.5 is not supported.")
    elif args.embed_model_name == "MedEmbed-small-v0.1":
        trainable_params = [
            "attention.self.query",
            "attention.self.key",
            "attention.self.value",
            "attention.output.dense",
            "intermediate.dense",
            "output.dense",
        ]
        # for name, param in base_model.named_parameters():
        #     print(name)
        # raise NotImplementedError("Fine-tuning MedEmbed-small-v0.1 is not supported.")
    else:
        raise ValueError(f"Invalid model name: {args.embed_model_name}")
    model = add_lora_to_model(
        model=base_model,
        target_modules=trainable_params,
        exclude_modules=exclude_modules,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    model.train()

    # Prepare the dataset
    corpuspath_2_corpus, inputpath_2_queries, inputpath_2_questiontype = {}, {}, {}
    inputpath_2_corpuspath = {}
    corpuspath_2_docid2context, corpuspath_2_query2positive = {}, {}
    inputpath_2_example_num = {}
    for corpuspath, inputpaths in args.corpuspath_2_inputpath.items():
        corpus = load_content(corpuspath)
        corpuspath_2_corpus[corpuspath] = corpus
        corpuspath_2_docid2context[corpuspath] = {chunk_dict["id"]: chunk_dict["origin_context"] for chunk_dict in corpus}

        for inputpath in inputpaths:
            chunk_dicts = load_content(inputpath)
            inputpath_2_queries[inputpath] = chunk_dicts
            inputpath_2_corpuspath[inputpath] = corpuspath
            if "content" in inputpath:
                question_type = "content"
            elif "entity_graph" in inputpath:
                question_type = "entity_graph"
            else:
                raise ValueError(f"Invalid input path: {inputpath}")
            inputpath_2_questiontype[inputpath] = question_type
            train_example_num = get_query_positive_map_from_contents(question_type, chunk_dicts, corpuspath_2_query2positive, corpuspath, args, model) # update corpuspath_2_query2positive
            inputpath_2_example_num[os.path.relpath(inputpath, CUSTOM_CORPUS_HOME)] = train_example_num
    all_train_contexts = {corpuspath: list(doc_dict.values()) for corpuspath, doc_dict in corpuspath_2_docid2context.items()}
    if rank == 0:
        logger.info(f"Training examples: {json.dumps(inputpath_2_example_num, indent=2)}")
        logger.info(f"Number of queries: {sum(inputpath_2_example_num.values())}")

    eval_corpuspath_2_corpus, eval_inputpath_2_queries, eval_inputpath_2_questiontype = {}, {}, {}
    eval_inputpath_2_corpuspath = {}
    for corpuspath, inputpaths in args.eval_corpuspath_2_inputpath.items():
        corpus = load_content(corpuspath)
        eval_corpuspath_2_corpus[corpuspath] = corpus

        for inputpath in inputpaths:
            chunk_dicts = load_content(inputpath)
            eval_inputpath_2_queries[inputpath] = chunk_dicts
            eval_inputpath_2_corpuspath[inputpath] = corpuspath
            if "content" in inputpath:
                question_type = "content"
            elif "entity_graph" in inputpath:
                question_type = "entity_graph"
            else:
                raise ValueError(f"Invalid input path: {inputpath}")
            eval_inputpath_2_questiontype[inputpath] = question_type

    # Prepare the data loader
    train_dataset = RetrievalDataset(
        corpuspath_2_query2positive=corpuspath_2_query2positive, 
        corpuspath_2_docid2context=corpuspath_2_docid2context
    )
    batch_sampler = StepWiseCorpusBatchSampler(
        corpuspath_2_indices=train_dataset.corpuspath_2_indices,
        batch_size=args.train_micro_batch_size_per_gpu,
        drop_last=True,  # Set to False if you want to include smaller batches
        seed=args.seed
    )
    train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler, collate_fn=custom_collate_fn)
    
    total_batches = len(batch_sampler)
    total_steps = args.num_epochs * ((total_batches - 1) // (args.train_micro_batch_size_per_gpu * args.gradient_accumulation_steps) + 1)
    if rank == 0:
        logger.info(f"Total Epochs: {args.num_epochs}; Each Rank Batch size: {args.train_micro_batch_size_per_gpu}; TRAIN_ACCUMULATION_STEPS: {args.gradient_accumulation_steps}; Total training steps: {total_steps}; Total training steps: {total_steps}")
    
    # Prepare Faiss index for negative sampling
    embedding_dim = model.get_sentence_embedding_dimension()
    corpuspath_2_index, corpuspath_2_id2docid, corpuspath_2_anceloss = {}, {}, {} # Each batch use same corpus
    with torch.no_grad():
        if rank == 0:
            logger.info("Building Faiss index for all documents...")
        # embed_index = tqdm(args.corpuspath_2_inputpath.keys(), desc = f"Embed Index", leave=True, position=1, disable=(rank != 0), dynamic_ncols=True)
        # for corpuspath in embed_index:
        for corpuspath in args.corpuspath_2_inputpath.keys():
            document_embeddings = model.encode(
                all_train_contexts[corpuspath],
                batch_size=args.query_batch_size,
                show_progress_bar=False,
                convert_to_tensor=True,
                device=device,
                num_workers=32,
                verbose=False
            ).float().cpu().numpy()
            if args.normalize_embeddings:
                document_embeddings = document_embeddings / np.linalg.norm(document_embeddings, ord=2, axis=1, keepdims=True)

            index = faiss.IndexFlatIP(embedding_dim)
            index.add(document_embeddings)
            corpuspath_2_index[corpuspath] = index
            
            corpuspath_2_id2docid[corpuspath] = {idx: doc_id for idx, doc_id in enumerate(corpuspath_2_docid2context[corpuspath].keys())}
            corpuspath_2_anceloss[corpuspath] = ANCELoss(
                model, 
                faiss_index = corpuspath_2_index[corpuspath],
                id_2_docid = corpuspath_2_id2docid[corpuspath], 
                query_2_positive = corpuspath_2_query2positive[corpuspath], 
                docid_2_context = corpuspath_2_docid2context[corpuspath],
                device=device,
                num_negatives=args.num_negatives, 
                loss_margin=args.loss_margin,
                normalize_embeddings=args.normalize_embeddings
            )
    
    if rank == 0:
        logger.info("Faiss index built.")

    # Prepare optimizer and scheduler
    to_be_optimized_params = []
    num_trainable_params, num_all_params = 0, 0
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
            num_trainable_params += param.numel()
            to_be_optimized_params.append(param)
        num_all_params += param.numel()
    if rank == 0:
        logger.info(f"Trainable parameters: {num_trainable_params}; All parameters: {num_all_params}, Ratio: {num_trainable_params / num_all_params * 100:.4f} %")
    
    pre_global_step = 0
    best_metric = -float('inf')  # Initialize best metric
    best_epoch, best_global_step = -1, -1
    if args.checkpoint_dir and args.resume_from_checkpoint:
        latest_checkpoint_path = get_latest_checkpoint_path(args.checkpoint_dir, rank)
        if latest_checkpoint_path:
            if rank == 0:
                logger.info(f"Loading checkpoint from {latest_checkpoint_path}")
            checkpoint = torch.load(latest_checkpoint_path, strict=False)  # Use strict=False due to LoRA adapters

            pre_global_step = checkpoint['global_step']
            args.log_dir = checkpoint['log_dir']
            if rank == 0:
                logger.info(f"Resumed training from checkpoint {args.checkpoint_dir} at global step {pre_global_step}, logging to {args.log_dir}")

            model.load_state_dict(checkpoint['model_state_dict'])
            
            optimizer = torch.optim.AdamW(to_be_optimized_params, lr=args.lr)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=total_steps
            )
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            random.setstate(checkpoint['random_state']['random'])
            np.random.set_state(checkpoint['random_state']['numpy'])
            torch.set_rng_state(checkpoint['random_state']['torch'])
            if torch.cuda.is_available():
                torch.cuda.set_rng_state_all(checkpoint['random_state']['torch_cuda'])
            
            best_metric = checkpoint.get('best_metric', best_metric)
            best_epoch = checkpoint.get('best_epoch', best_epoch)
            best_global_step = checkpoint.get('best_global_step', best_global_step)
        else:
            if rank == 0:
                logger.warning("No checkpoint found. Starting from scratch.")
            pre_global_step = 0
            
            optimizer = torch.optim.AdamW(to_be_optimized_params, lr=args.lr)
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=total_steps
            )
    else:
        pre_global_step = 0
        if rank == 0:
            logger.info(f"Training from scratch, logging to {args.log_dir}")
        optimizer = torch.optim.AdamW(to_be_optimized_params, lr=args.lr)
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_steps
        )
    
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )

    # Prepare TensorBoard writer (only on rank 0)
    if rank == 0:
        writer = SummaryWriter(log_dir=args.log_dir)

    if args.do_eval:
        model.eval()
        with torch.no_grad():
            question_categories = ["original-question", "rephrased-questions", "rephrased-questions-part", "rephrased-questions-hybrid"]
            all_metric_dicts = {}
            # eval_file_pbar = tqdm(eval_inputpath_2_queries.items(), desc=f"Eval File", leave=False, position=1, disable=(rank != 0), dynamic_ncols=True)
            # for inputpath, test_queries in eval_file_pbar:
            for inputpath, test_queries in eval_inputpath_2_queries.items():
                
                # generate the retrieval results
                contexts_to_process = [cur_dict["origin_context"] for cur_dict in eval_corpuspath_2_corpus[eval_inputpath_2_corpuspath[inputpath]]]
                all_embeddings = calculate_all_embeddings(contexts_to_process, model, args.normalize_embeddings, args.add_eos, args.query_batch_size)

                # generate the retrieval results
                chunkid_2_dict = {i: cur_dict for i, cur_dict in enumerate(eval_corpuspath_2_corpus[eval_inputpath_2_corpuspath[inputpath]])}
                question_type = eval_inputpath_2_questiontype[inputpath]
                for question_category in question_categories:
                    process_questions(test_queries, question_type, question_category, model, all_embeddings, chunkid_2_dict, args)
                
                # calculate the metrics
                if question_type == "content":
                    metric_dicts = eval_retrieval_results_for_file_content(args, test_queries, max_process_num=args.max_eval_num)
                elif question_type == "entity_graph":
                    metric_dicts = eval_retrieval_results_for_file_entity_graph(args, test_queries, max_process_num=args.max_eval_num)
                else:
                    raise ValueError(f"Invalid question type: {question_type}")
                metric_dicts = {
                    # "Precision@3; full; transformation@0": metric_dicts["Precision@3; full; transformation@0"],
                    # "Precision@3; full; transformation@4": metric_dicts["Precision@3; full; transformation@7"],
                    # "Precision@3; part; transformation@0": metric_dicts["Precision@3; part; transformation@0"],
                    # "Precision@3; part; transformation@3": metric_dicts["Precision@3; part; transformation@7"],
                    "Precision@3; hybrid; transformation@0": metric_dicts["Precision@3; hybrid; transformation@0"],
                    "Precision@3; hybrid; transformation@7": metric_dicts["Precision@3; hybrid; transformation@7"],
                }
                all_metric_dicts[inputpath] = metric_dicts

            if rank == 0:
                # logger.info(f"Metrics: {json.dumps(all_metric_dicts, indent=2)}")
                for inputpath, metric_dicts in all_metric_dicts.items():
                    for key, value in metric_dicts.items():
                        writer.add_scalar(f"{key}/{inputpath}", float(value), -1)

    # Start training
    epoch_pbar = tqdm(range(args.num_epochs), desc="Epochs", leave=True, position=1, disable=(rank != 0), dynamic_ncols=True)
    global_step = pre_global_step
    with accelerator.autocast():
        for epoch in epoch_pbar:
            
            break_step_or_not = False
            epoch_loss = 0.0
            batch_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}", unit="batch", leave=True, position=1, disable=(rank != 0), dynamic_ncols=True)
            for step_idx, batch in enumerate(batch_pbar, start=1):
                if (epoch * total_batches + step_idx) < pre_global_step:
                    continue
                with accelerator.accumulate(model):

                    queries = batch['query']  # List of strings
                    positives = batch['positives']  # List of lists
                    corpuspath = batch['corpuspaths'][0]  # Single string

                    max_positives = max(len(p) for p in positives) # Find the maximum number of positive samples in the batch
                    padded_positives = [p + ['[PAD]'] * (max_positives - len(p)) for p in positives] # Pad the positive lists, using empty strings as the padding value
                    masks = [[1] * len(p) + [0] * (max_positives - len(p)) for p in positives] # Create masks: 1 represents a valid positive sample, 0 represents padding
                    masks = torch.tensor(masks, dtype=torch.bool, device=device)

                    ance_loss = corpuspath_2_anceloss[corpuspath](queries, padded_positives, masks)
                    accelerator.backward(ance_loss)

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    if rank == 0:
                        writer.add_scalar('Loss/train_ANCE', ance_loss.item(), global_step)
                        writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], global_step)
                        batch_pbar.set_postfix({"global_step": f"{global_step}", "loss": f"{ance_loss.item():.4f}"})
                    
                    global_step += 1
                    epoch_loss += ance_loss.item()

                    if args.checkpoint_dir and (global_step % args.checkpoint_freq == 0):
                        checkpoint_state = {
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'global_step': global_step,
                            'log_dir': args.log_dir,
                            'random_state': {
                                    'random': random.getstate(),
                                    'numpy': np.random.get_state(),
                                    'torch': torch.get_rng_state(),
                                    'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                                },
                            'hyperparameters': vars(args),
                            'best_metric': best_metric,
                            'best_epoch': best_epoch,
                            'best_global_step': best_global_step
                        }
                        save_checkpoint(checkpoint_state, args.checkpoint_dir, global_step, epoch, max_checkpoints=args.max_checkpoints, rank=rank)
                    
                    if args.do_eval and (global_step % args.eval_steps == 0):
                        model.eval()
                        with torch.no_grad():
                            question_categories = ["rephrased-questions-hybrid"]
                            # question_categories = ["original-question", "rephrased-questions", "rephrased-questions-part", "rephrased-questions-hybrid"]
                            all_metric_dicts = {}
                            # eval_file_pbar = tqdm(eval_inputpath_2_queries.items(), desc=f"Eval File", leave=True, position=1, disable=(rank != 0), dynamic_ncols=True)
                            # for inputpath, test_queries in eval_file_pbar:
                            for inputpath, test_queries in eval_inputpath_2_queries.items():
                                
                                # generate the retrieval results
                                contexts_to_process = [cur_dict["origin_context"] for cur_dict in eval_corpuspath_2_corpus[eval_inputpath_2_corpuspath[inputpath]]]
                                all_embeddings = calculate_all_embeddings(contexts_to_process, model, args.normalize_embeddings, args.add_eos, args.query_batch_size)

                                # generate the retrieval results
                                chunkid_2_dict = {i: cur_dict for i, cur_dict in enumerate(eval_corpuspath_2_corpus[eval_inputpath_2_corpuspath[inputpath]])}
                                question_type = eval_inputpath_2_questiontype[inputpath]
                                for question_category in question_categories:
                                    process_questions(test_queries, question_type, question_category, model, all_embeddings, chunkid_2_dict, args)
                                
                                # calculate the metrics
                                if question_type == "content":
                                    metric_dicts = eval_retrieval_results_for_file_content(args, test_queries, max_process_num=args.max_eval_num)
                                elif question_type == "entity_graph":
                                    metric_dicts = eval_retrieval_results_for_file_entity_graph(args, test_queries, max_process_num=args.max_eval_num)
                                else:
                                    raise ValueError(f"Invalid question type: {question_type}")
                                metric_dicts = {
                                    # "Precision@3; full; transformation@0": metric_dicts["Precision@3; full; transformation@0"],
                                    # "Precision@3; full; transformation@4": metric_dicts["Precision@3; full; transformation@7"],
                                    # "Precision@3; part; transformation@0": metric_dicts["Precision@3; part; transformation@0"],
                                    # "Precision@3; part; transformation@3": metric_dicts["Precision@3; part; transformation@7"],
                                    "Precision@3; hybrid; transformation@0": metric_dicts["Precision@3; hybrid; transformation@0"],
                                    "Precision@3; hybrid; transformation@7": metric_dicts["Precision@3; hybrid; transformation@7"],
                                }
                                all_metric_dicts[inputpath] = metric_dicts

                            # Extract the primary metric from the first inputpath
                            if eval_inputpath_2_queries:
                                first_eval_inputpath = next(iter(eval_inputpath_2_queries))
                                primary_metric = float(all_metric_dicts[first_eval_inputpath].get("Precision@3; hybrid; transformation@7", 0.0))
                            else:
                                primary_metric = 0.0
                            
                            # Check if the current metric is better than the best
                            if primary_metric > best_metric:
                                best_metric = primary_metric
                                best_epoch = epoch  # Assuming epoch starts at 0
                                best_global_step = global_step - 1
                                if rank == 0:
                                    logger.info(f"New best metric {best_metric} achieved at epoch {best_epoch}, global_step {best_global_step}. Saving best checkpoint.")
                                # checkpoint_state = {
                                #     'epoch': epoch,
                                #     'model_state_dict': model.state_dict(),
                                #     'optimizer_state_dict': optimizer.state_dict(),
                                #     'scheduler_state_dict': scheduler.state_dict(),
                                #     'global_step': global_step,
                                #     'log_dir': args.log_dir,
                                #     'random_state': {
                                #             'random': random.getstate(),
                                #             'numpy': np.random.get_state(),
                                #             'torch': torch.get_rng_state(),
                                #             'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                                #         },
                                #     'hyperparameters': vars(args),
                                #     'best_metric': best_metric,
                                #     'best_epoch': best_epoch,
                                #     'best_global_step': best_global_step
                                # }
                                # save_checkpoint(checkpoint_state, args.checkpoint_dir, global_step, epoch, max_checkpoints=args.max_checkpoints, rank=rank, is_best=True)

                            if rank == 0:
                                # logger.info(f"Metrics: {json.dumps(all_metric_dicts, indent=2)}")
                                for inputpath, metric_dicts in all_metric_dicts.items():
                                    for key, value in metric_dicts.items():
                                        writer.add_scalar(f"{key}/{inputpath}", float(value), global_step)
                        model.train()

                    if global_step % args.faiss_update_freq == 0:
                        with torch.no_grad():
                            # embed_index = tqdm(args.corpuspath_2_inputpath.keys(), desc = f"Embed Index", leave=False, position=1, disable=(rank != 0), dynamic_ncols=True)
                            # for corpuspath in embed_index:
                            for corpuspath in args.corpuspath_2_inputpath.keys():
                                # print(f"Rank {rank} updating Faiss index for {corpuspath}")
                                # beg_time = time.time()
                                document_embeddings = model.encode(
                                    all_train_contexts[corpuspath],
                                    batch_size=args.query_batch_size,
                                    show_progress_bar=False,
                                    convert_to_tensor=True,
                                    device=device,
                                    num_workers=32,
                                    verbose=False
                                ).float().cpu().numpy()
                                # print(f"Rank {rank} updated Faiss index for {corpuspath} in {time.time() - beg_time:.2f} seconds")

                                # beg_time = time.time()
                                if args.normalize_embeddings:
                                    document_embeddings = document_embeddings / np.linalg.norm(document_embeddings, ord=2, axis=1, keepdims=True)
                                # print(f"Rank {rank} normalized Faiss index for {corpuspath} in {time.time() - beg_time:.2f} seconds")

                                # beg_time = time.time()
                                corpuspath_2_index[corpuspath].reset()
                                corpuspath_2_index[corpuspath].add(document_embeddings)
                                # print(f"Rank {rank} added Faiss index for {corpuspath} in {time.time() - beg_time:.2f} seconds")

                                # exit(0)

                if args.max_train_steps and global_step >= args.max_train_steps:
                    break_step_or_not = True
                    break
            
            if break_step_or_not:
                if args.do_eval:
                    model.eval()
                    with torch.no_grad():
                        question_categories = ["rephrased-questions-hybrid"]
                        # question_categories = ["original-question", "rephrased-questions", "rephrased-questions-part", "rephrased-questions-hybrid"]
                        all_metric_dicts = {}
                        # eval_file_pbar = tqdm(eval_inputpath_2_queries.items(), desc=f"Eval File", leave=True, position=1, disable=(rank != 0), dynamic_ncols=True)
                        # for inputpath, test_queries in eval_file_pbar:
                        for inputpath, test_queries in eval_inputpath_2_queries.items():
                            
                            # generate the retrieval results
                            contexts_to_process = [cur_dict["origin_context"] for cur_dict in eval_corpuspath_2_corpus[eval_inputpath_2_corpuspath[inputpath]]]
                            all_embeddings = calculate_all_embeddings(contexts_to_process, model, args.normalize_embeddings, args.add_eos, args.query_batch_size)

                            # generate the retrieval results
                            chunkid_2_dict = {i: cur_dict for i, cur_dict in enumerate(eval_corpuspath_2_corpus[eval_inputpath_2_corpuspath[inputpath]])}
                            question_type = eval_inputpath_2_questiontype[inputpath]
                            for question_category in question_categories:
                                process_questions(test_queries, question_type, question_category, model, all_embeddings, chunkid_2_dict, args)
                            
                            # calculate the metrics
                            if question_type == "content":
                                metric_dicts = eval_retrieval_results_for_file_content(args, test_queries, max_process_num=args.max_eval_num)
                            elif question_type == "entity_graph":
                                metric_dicts = eval_retrieval_results_for_file_entity_graph(args, test_queries, max_process_num=args.max_eval_num)
                            else:
                                raise ValueError(f"Invalid question type: {question_type}")
                            metric_dicts = {
                                    # "Precision@3; full; transformation@0": metric_dicts["Precision@3; full; transformation@0"],
                                    # "Precision@3; full; transformation@4": metric_dicts["Precision@3; full; transformation@7"],
                                    # "Precision@3; part; transformation@0": metric_dicts["Precision@3; part; transformation@0"],
                                    # "Precision@3; part; transformation@3": metric_dicts["Precision@3; part; transformation@7"],
                                    "Precision@3; hybrid; transformation@0": metric_dicts["Precision@3; hybrid; transformation@0"],
                                    "Precision@3; hybrid; transformation@7": metric_dicts["Precision@3; hybrid; transformation@7"],
                                }
                            all_metric_dicts[inputpath] = metric_dicts

                        # Extract the primary metric from the first inputpath
                        if eval_inputpath_2_queries:
                            first_eval_inputpath = next(iter(eval_inputpath_2_queries))
                            primary_metric = float(all_metric_dicts[first_eval_inputpath].get("Precision@3; hybrid; transformation@7", 0.0))
                        else:
                            primary_metric = 0.0
                        
                        # Check if the current metric is better than the best
                        if primary_metric > best_metric:
                            best_metric = primary_metric
                            best_epoch = epoch  # Assuming epoch starts at 0
                            best_global_step = global_step - 1
                            if rank == 0:
                                logger.info(f"New best metric {best_metric} achieved at epoch {best_epoch}, global_step {best_global_step}. Saving best checkpoint.")
                            # checkpoint_state = {
                            #     'epoch': epoch,
                            #     'model_state_dict': model.state_dict(),
                            #     'optimizer_state_dict': optimizer.state_dict(),
                            #     'scheduler_state_dict': scheduler.state_dict(),
                            #     'global_step': global_step,
                            #     'log_dir': args.log_dir,
                            #     'random_state': {
                            #             'random': random.getstate(),
                            #             'numpy': np.random.get_state(),
                            #             'torch': torch.get_rng_state(),
                            #             'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                            #         },
                            #     'hyperparameters': vars(args),
                            #     'best_metric': best_metric,
                            #     'best_epoch': best_epoch,
                            #     'best_global_step': best_global_step
                            # }
                            # save_checkpoint(checkpoint_state, args.checkpoint_dir, global_step, epoch, max_checkpoints=args.max_checkpoints, rank=rank, is_best=True)

                        if rank == 0:
                            # logger.info(f"Metrics: {json.dumps(all_metric_dicts, indent=2)}")
                            for inputpath, metric_dicts in all_metric_dicts.items():
                                for key, value in metric_dicts.items():
                                    writer.add_scalar(f"{key}/{inputpath}", float(value), global_step)
                    model.train()
                
                break

            if rank == 0:
                avg_epoch_loss = epoch_loss / len(train_dataloader)
                logger.info(f"Epoch {epoch} completed. Average Loss: {avg_epoch_loss:.4f}")

            batch_sampler.reset()

    accelerator.wait_for_everyone()

    if args.output_model_dir:
        if rank == 0:
            # Save the latest checkpoint
            final_checkpoint_path = os.path.join(args.output_model_dir, f"final_latest_global_step_{global_step}_epoch_{args.num_epochs}.pt")
            torch.save({
                'epoch': args.num_epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'global_step': global_step,
                'log_dir': args.log_dir,
                'random_state': {
                    'random': random.getstate(),
                    'numpy': np.random.get_state(),
                    'torch': torch.get_rng_state(),
                    'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                },
                'hyperparameters': vars(args),
                'best_metric': best_metric,
                'best_epoch': best_epoch,
                'best_global_step': best_global_step
            }, final_checkpoint_path)
            logger.info(f"Final latest checkpoint saved to {final_checkpoint_path}")

            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(args.output_model_dir)
    
    if rank == 0:
        writer.close()

if __name__ == "__main__":
    main()
