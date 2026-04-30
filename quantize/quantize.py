import torch
import os
from model.decoder_transformer import decoder_transformer
from torch import nn

device = 'cpu'
model_path = 'ckpt/sft_epoch_100.pth' 
quantized_model_path = 'ckpt/sft_100_q.pth' 

print(f"正在从 {model_path} 加载模型...")
checkpoint = torch.load(model_path, map_location=device)
vocab_size = 100277
d_model = 512
n_layer = 8
n_head = 8
d_hidden = d_model * 2

pre_model = decoder_transformer(vocab_size, d_model, n_layer, n_head, d_hidden).to(device)
state_dict = checkpoint
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
pre_model.load_state_dict(state_dict)
pre_model.to(device)
pre_model.eval()
print("原始模型加载成功。")

print("\n开始对模型的 nn.Linear 层进行动态量化...")
quantized_model = torch.quantization.quantize_dynamic(
    pre_model,
    {nn.Linear},
    dtype=torch.qint8
)
print("动态量化完成。")
torch.save(quantized_model.state_dict(), quantized_model_path)
print(f"量化后的模型已保存到 {quantized_model_path}")


print(f"\n--- 1. 全量参数体积对比 ---")
original_state_dict_path = 'ckpt/original_state_dict.pth'
torch.save(pre_model.state_dict(), original_state_dict_path)
original_size = os.path.getsize(original_state_dict_path)
quantized_size = os.path.getsize(quantized_model_path)

if quantized_size > 0:
    compression_ratio = original_size / quantized_size
    print(f"原始模型 state_dict 大小: {original_size / 1e6:.2f} MB")
    print(f"量化模型 state_dict 大小: {quantized_size / 1e6:.2f} MB")
    print(f"实际总压缩比: {compression_ratio:.2f} : 1")
else:
    print("无法计算总压缩比。")
os.remove(original_state_dict_path)
