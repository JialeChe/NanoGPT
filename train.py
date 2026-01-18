import torch
from dataset.dataloader import create_dataloader
from model.decoder_transformer import decoder_transformer
import torch.optim as optim
import torch.nn as nn
import os
import tiktoken
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def train_model(data_path, block_size, batch_size, vocab_size,
                d_model, n_layer, n_head, d_hidden,
                learning_rate, num_epochs, device,t_writer):
    
    print("Creating dataloader...")
    train_loader, vocab_size = create_dataloader(data_path, block_size, batch_size)
    
    model = decoder_transformer(vocab_size, d_model, n_layer, n_head, d_hidden).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    cos_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        writer = SummaryWriter(f'runs/epoch_{epoch}_loss')
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)  # [B,L,vocab_size]
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            writer.add_scalar('Loss/train', loss.item(), batch_idx)
            pbar.set_postfix({'loss': f'{loss.item():.4f}','total_loss': f'{total_loss:.4f}'})
        torch.save(model.state_dict(), f'ckpt/model_epoch_{epoch+1}.pth')
        cos_scheduler.step()
        avg_loss = total_loss / len(train_loader)
        t_writer.add_scalar('Loss/train', avg_loss, epoch)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        writer.close()
    
    return model

def generate_story(model, start_string, max_len, device, block_size):
    model.eval()
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    # Encode the starting string into token IDs
    start_ids = tokenizer.encode(start_string)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    # Start generating
    with torch.no_grad():
        for _ in range(max_len):
            x_cond = x if x.size(1) <= block_size else x[:, -block_size:]
            logits = model(x_cond)
            logits = logits[:, -1, :] # Shape (B, C)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, idx_next), dim=1)

    generated_text = tokenizer.decode(x[0].tolist())
    print("\n--- Generated Story ---\n")
    print(generated_text)
    print("\n------------------\n")


if __name__ == '__main__':
    # --- h-paraments setting ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, 'dataset/TinyStoriesV2-GPT4-train.txt') 
    
    block_size = 256
    batch_size = 16
    vocab_size = 100277
    
    d_model = 512
    n_layer = 8
    n_head = 8
    d_hidden = d_model * 2
    
    learning_rate = 1e-4
    num_epochs = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device using: {device}")

    # --- train ---
    if os.path.exists(data_path):
        print("strat training...")
        t_writer = SummaryWriter(f'runs/total_loss')
        trained_model = train_model(data_path, block_size, batch_size, vocab_size,
                                    d_model, n_layer, n_head, d_hidden,
                                    learning_rate, num_epochs, device,t_writer)
        
        writer.close()
        
        # # --- test ---
        # generate_story(trained_model, 
        #                start_string="Once upon a time", 
        #                max_len=100, 
        #                device=device,
        #                block_size=block_size)
    else:
        print(f"error: data file not found {data_path}")
        print("please ensure the file 'dataset/TinyStoriesV2-GPT4-train.txt' exists.")