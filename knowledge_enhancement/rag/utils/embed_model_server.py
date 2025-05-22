from sentence_transformers import SentenceTransformer
import logging
import torch
import queue
import threading
import argparse
from flask import Flask, request, jsonify

httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Calculate embeddings.")
parser.add_argument("--embed_model_path", type=str, required=True, help="Path to the embed_model directory.")
parser.add_argument("--max_seq_length", type=int, help="Maximum sequence length.")
parser.add_argument("--port", type=int, default=6001, help="Port number.")
args = parser.parse_args()

app = Flask(__name__)

def load_sentence_transformer_model(embed_model_path, max_seq_length=32768):
    """
    Load and configure the SentenceTransformer embed_model based on the embed_model path.
    """
    logging.getLogger().setLevel(logging.WARNING)

    if any(name in embed_model_path for name in [
        "NV-Embed-v2", "MiniCPM-Embedding",
        "snowflake-arctic-embed-m-v1.5", "MedEmbed-small-v0.1"
    ]):
        embed_model = SentenceTransformer(
            embed_model_path, trust_remote_code=True,
            model_kwargs={"torch_dtype": torch.float16}
        )
    elif any(name in embed_model_path for name in [
        "stella_en_1.5B_v5", "stella_en_400M_v5"
    ]):
        embed_model = SentenceTransformer(embed_model_path, trust_remote_code=True)
    else:
        raise ValueError(f"Model {embed_model_path} is not supported.")

    # Set maximum sequence length
    embed_model.max_seq_length = max_seq_length
    return embed_model

# Create a global request queue
request_queue = queue.Queue()
processing_lock = threading.Lock()
embed_model = load_sentence_transformer_model(args.embed_model_path, args.max_seq_length)

def process_requests():
    while True:
        # Retrieve a request from the queue
        item = request_queue.get()
        if item == None:
            break  # Termination signal received, exit loop
        data, res = item  # 'data' is the extracted JSON data
        with processing_lock:
            queries = data.get('queries')
            batch_size = data.get('batch_size')
            prompt = data.get('query_prefix')
            prompt_name = data.get('prompt_name')
            normalize_embeddings = data.get('normalize_embeddings')
            show_progress_bar = data.get('show_progress_bar')

            if not queries or not batch_size or not prompt or not prompt_name or not normalize_embeddings:
                response_data = {"error": "Missing required parameters"}
                status_code = 400 
            else:
                query_embeddings = embed_model.encode(queries, batch_size=batch_size, show_progress_bar=show_progress_bar)
                response_data = {"query_embeddings": probabilities}
                status_code = 200

            # Place the result in the response queue
            res.put((response_data, status_code))
        request_queue.task_done()

# Start the request processing thread
processing_thread = threading.Thread(target=process_requests, daemon=True)
processing_thread.start()

@app.route('/calculate_embeddings', methods=['POST'])
def calculate_embeddings():
    # Extract data from the request within the request context
    data = request.get_json()
    # Create a response queue for each request
    response_queue = queue.Queue()
    # Place the data and response queue into the global request queue
    request_queue.put((data, response_queue))
    # Wait for processing to complete and get the response
    response_data, status_code = response_queue.get()
    return jsonify(response_data), status_code

app.run(host='0.0.0.0', port=args.port)