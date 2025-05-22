import os
import argparse
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

def main(checkpoint_dir):
    global_step_dir = os.path.join(checkpoint_dir, "global_step200")
    output_dir = os.path.join(checkpoint_dir)

    # 使用 DeepSpeed 提供的脚本将 Zero 3 优化的分片合并成一个完整的模型
    deepspeed_script = os.path.join(checkpoint_dir, "zero_to_fp32.py")

    # 合并分片到一个完整的模型文件
    os.system(f"python {deepspeed_script} {checkpoint_dir} {output_dir}")

    # 验证模型是否可以正确加载
    model = AutoModelForCausalLM.from_pretrained(output_dir)
    
    print("Model successfully converted and loaded.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DeepSpeed checkpoint to Hugging Face format.")
    parser.add_argument("checkpoint_dir", type=str, help="Path to the checkpoint directory")

    args = parser.parse_args()
    main(args.checkpoint_dir)
