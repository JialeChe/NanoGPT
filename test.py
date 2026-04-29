import torch
import os
import tiktoken
import argparse

from checkpoint_utils import extract_state_dict, resolve_model_path
from config_utils import create_model_from_config, get_block_size, get_default_config_path, load_experiment_config


def test_full_training(model, start_string, max_len, device, block_size,tokenizer):
    model.eval()
    start_ids = tokenizer.encode(start_string)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    with torch.no_grad():
        for _ in range(max_len):
            x_cond = x if x.size(1) <= block_size else x[:, -block_size:]
            logits = model(x_cond)
            logits = logits[:, -1, :] # Shape (B, C)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, idx_next), dim=1)
    generated_ids = x[0].tolist()
    generated_text = tokenizer.decode(generated_ids)
    generated_text = generated_text.split("<|endoftext|>")[0]
    print(generated_text)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--ckpt-dir", default=None)
    parser.add_argument("--model-path", default=None)
    args = parser.parse_args()

    config_path = args.config or get_default_config_path(current_dir, sft=False)
    config = load_experiment_config(config_path, current_dir)
    checkpoint_dir = args.ckpt_dir or config["checkpoint"]["dir"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = resolve_model_path(args.model_path or config.get("inference", {}).get("model_path"), checkpoint_dir, prefer_sft=False)

    model = create_model_from_config(config).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(extract_state_dict(checkpoint))

    start_string = "I love Yina Wang.She is a beautiful girl."
    max_len = 512
    block_size = get_block_size(config)
    tokenizer = tiktoken.get_encoding("cl100k_base")

    test_full_training(model, start_string, max_len, device, block_size,tokenizer)