import torch
from torch.utils.data import Dataset, DataLoader
import os
import tiktoken
import random
import json

class StoryDataset(Dataset):
    def __init__(self, data_path, block_size):
        self.block_size = block_size
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.vocab_size = self.tokenizer.n_vocab
        self.eot_token = self.tokenizer.eot_token

        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        self.stories = [s.strip() for s in text.split('<|endoftext|>') if s.strip()]

    def __len__(self):
        return len(self.stories)

    def __getitem__(self, idx):
        story_text = self.stories[idx]
        tokens = self.tokenizer.encode(story_text)
        if len(tokens) < self.block_size + 1:
            tokens += [self.eot_token] * (self.block_size + 1 - len(tokens))
        else:
            start_idx = random.randint(0, len(tokens) - (self.block_size + 1))
            tokens = tokens[start_idx:start_idx + self.block_size + 1]
            
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y

class sft_dataset(Dataset):
    def __init__(self, data_path, max_len):
        self.max_len = max_len
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.vocab_size = self.tokenizer.n_vocab
        self.eot_token = self.tokenizer.eot_token
        self.pad_id = -100

        self.sft_data = json.load(open(data_path, 'r', encoding='utf-8'))

    def __len__(self):
        return len(self.sft_data)

    def __getitem__(self, idx):
        item = self.sft_data[idx]
        instruction = item['instruction']
        input_ = item.get('input', '')
        output = item.get('output', '')

        if input_ is not None:
            text = "指令:" + instruction + "\n输入:" + input_ + "\n输出:"
        else:
            text = "指令:" + instruction + "\n输出:"

        full = text + output

        prompt_tokens = self.tokenizer.encode(text)
        full_tokens = self.tokenizer.encode(full)
        labels = [self.pad_id] * len(prompt_tokens) + full_tokens[len(prompt_tokens):]
        if len(full_tokens) < self.max_len + 1:
            pad_length = self.max_len + 1 - len(full_tokens)
            full_tokens += [self.eot_token] * pad_length
            labels += [self.pad_id] * pad_length
        else:
            start =  random.randint(0, len(full_tokens) - self.max_len - 1)
            full_tokens = full_tokens[start : start + self.max_len+1]
            labels = labels[start : start + self.max_len+1]
        print(len(full_tokens), len(labels))
        x = torch.tensor(full_tokens[:-1], dtype=torch.long)
        y = torch.tensor(labels[1:], dtype=torch.long)

        return x, y

def create_dataloader(data_path, block_size, batch_size, shuffle=True, num_workers=4, pin_memory=True,sft=False):
    if sft:
        dataset = sft_dataset(data_path, max_len=block_size)
    else:
        dataset = StoryDataset(data_path, block_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory, drop_last=True)
    return dataloader, dataset.vocab_size



if __name__ == '__main__':
    train_data_path = 'sft_data/sft_1500.json'
    
    block_size = 256  
    batch_size = 64   

    if os.path.exists(train_data_path):
        print(f"从 {train_data_path} 创建 DataLoader...")
        train_loader, vocab_size = create_dataloader(train_data_path, block_size, batch_size,sft=True)
        
        print(f"故事总数: {len(train_loader.dataset)}")
        print(f"词汇表大小: {vocab_size}")
        print(f"每个epoch的batch数量: {len(train_loader)}")

        for i, (x, y) in enumerate(train_loader):
            print(f"\n--- Batch {i+1} ---")
            print("x shape:", x.shape)
            print("y shape:", y.shape)
            assert x.shape == (batch_size, block_size)
            assert y.shape == (batch_size, block_size)
            # assert torch.equal(x[:, 1:], y[:, :-1])
            
            if i == 0:
                print("x[0] (前10个 tokens):", x[0, 50:])
                print("y[0] (前10个 tokens):", y[0, 49:])
                break

    else:
        print(f"错误: 找不到数据文件 {train_data_path}")