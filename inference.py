import torch
import tiktoken
import argparse
import os
from checkpoint_utils import extract_state_dict, resolve_model_path
from config_utils import create_model_from_config, get_block_size, get_default_config_path, load_experiment_config

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--ckpt-dir", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    config_path = args.config or get_default_config_path(current_dir, sft=True)
    config = load_experiment_config(config_path, current_dir)
    inference_config = config.get("inference", {})
    checkpoint_config = config.get("checkpoint", {})

    device = args.device or inference_config.get("device") or ('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_dir = args.ckpt_dir if args.ckpt_dir is not None else checkpoint_config["dir"]
    configured_model_path = args.model_path if args.model_path is not None else inference_config.get("model_path")
    model_path = resolve_model_path(configured_model_path, checkpoint_dir, prefer_sft=True)
    is_quantized = 'quantized' in os.path.basename(model_path)
    if is_quantized:
        device = 'cpu'
        print("检测到量化模型，将强制使用 CPU。")

    block_size = get_block_size(config)
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else inference_config.get("max_new_tokens", 1024)
    temperature = args.temperature if args.temperature is not None else inference_config.get("temperature", 0.9)
    top_k = args.top_k if args.top_k is not None else inference_config.get("top_k", 1)

    print(f"正在从 {model_path} 加载模型...")
    model = create_model_from_config(config)
    if is_quantized:
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = extract_state_dict(checkpoint)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("模型加载成功！")
    tokenizer = tiktoken.get_encoding("cl100k_base")
    print("输入 'exit' 或 'quit' 退出程序。")

    while True:
        instruction = input("请输入指令 (instruction): ")
        if instruction.lower() in ['exit', 'quit']:
            break
        user_input = input("请输入输入 (input, 可选, 直接回车跳过): ")
        if user_input:
            prompt = f"指令:{instruction}\n输入:{user_input}\n输出:"
        else:
            prompt = f"指令:{instruction}\n输出:"
            
        print("\n[构建的 Prompt]")
        print(prompt, end='')
        start_ids = tokenizer.encode(prompt)
        x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

        with torch.no_grad():
            for _ in range(max_new_tokens):
                x_cond = x if x.size(1) <= block_size else x[:, -block_size:]
                logits = model(x_cond)
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = torch.nn.functional.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                if idx_next == tokenizer.eot_token:
                    break
                x = torch.cat((x, idx_next), dim=1)
        output_ids = x[0][len(start_ids):].tolist()
        output_text = tokenizer.decode(output_ids)
        print(output_text)
        print("-" * 30)


if __name__ == "__main__":
    main()
