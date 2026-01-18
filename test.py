import torch
from dataset.dataloader import create_dataloader
from model.decoder_transformer import decoder_transformer
import torch.optim as optim
import torch.nn as nn
import os
import tiktoken


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
    # 过滤掉<|endoftext|>
    generated_text = generated_text.split("<|endoftext|>")[0]
    print(generated_text)

if __name__ == "__main__":
    model = decoder_transformer(vocab_size=100277,d_model=512,n_layer=8,n_head=8,d_hidden=1024,
                                dropout=0.1,use_moe=True,n_expert=4,top_k=2,max_seq_len=512).cuda()
    model.load_state_dict(torch.load("ckpt/model_epoch_10.pth"))

    start_string = "I love Yina Wang.She is a beautiful girl."
    max_len = 512
    device = "cuda"
    block_size = 256
    tokenizer = tiktoken.get_encoding("cl100k_base")

    test_full_training(model, start_string, max_len, device, block_size,tokenizer)