import torch
import tiktoken
from model.decoder_transformer import decoder_transformer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = 'ckpt/sft_epoch_91.pth'
is_quantized = 'quantized' in model_path
if is_quantized:
    device = 'cpu'
    print("检测到量化模型，将强制使用 CPU。")

vocab_size = 100277 
d_model = 512
n_layer = 8
n_head = 8
d_hidden = d_model * 2
block_size = 256

max_new_tokens = 1024
temperature = 0.9
top_k = 1
print(f"正在从 {model_path} 加载模型...")
model = decoder_transformer(vocab_size, d_model, n_layer, n_head, d_hidden)
if is_quantized:
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
checkpoint = torch.load(model_path, map_location=device)
state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint

unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

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
