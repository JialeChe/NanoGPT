import torch
from dataset.dataloader import create_dataloader
from utils.checkpoint_utils import (
    PRETRAIN_STAGE,
    SFT_STAGE,
    build_checkpoint_path,
    ensure_checkpoint_dirs,
    find_latest_checkpoint,
)
from utils.config_utils import create_model_from_config, get_block_size, get_default_config_path, load_experiment_config
import torch.optim as optim
import torch.nn as nn
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse

def train_model(config, data_path,
                learning_rate, num_epochs, device, t_writer,
                checkpoint_dir, save_every):
    block_size = get_block_size(config)
    batch_size = config["data"]["batch_size"]
    shuffle = config["data"].get("shuffle", True)
    num_workers = config["data"].get("num_workers", 4)
    pin_memory = config["data"].get("pin_memory", True)

    print("Creating dataloader...")
    train_loader, vocab_size = create_dataloader(
        data_path,
        block_size,
        batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    ensure_checkpoint_dirs(checkpoint_dir)

    if vocab_size != config["model"]["vocab_size"]:
        raise ValueError(f"Config vocab_size {config['model']['vocab_size']} does not match dataset vocab_size {vocab_size}")

    model = create_model_from_config(config).to(device)
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
        if (epoch + 1) % save_every == 0:
            torch.save(model.state_dict(), build_checkpoint_path(checkpoint_dir, PRETRAIN_STAGE, epoch + 1))
        cos_scheduler.step()
        avg_loss = total_loss / len(train_loader)
        t_writer.add_scalar('Loss/train', avg_loss, epoch)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        writer.close()
    

def sft_train(config, data_path,
                learning_rate, num_epochs, device, t_writer,
                checkpoint_dir, pretrained_model_path, save_every):
    block_size = get_block_size(config)
    batch_size = config["data"]["batch_size"]
    shuffle = config["data"].get("shuffle", True)
    num_workers = config["data"].get("num_workers", 4)
    pin_memory = config["data"].get("pin_memory", True)

    print("Creating dataloader...")
    train_loader, vocab_size = create_dataloader(
        data_path,
        block_size,
        batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        sft=True,
    )
    ensure_checkpoint_dirs(checkpoint_dir)

    if vocab_size != config["model"]["vocab_size"]:
        raise ValueError(f"Config vocab_size {config['model']['vocab_size']} does not match dataset vocab_size {vocab_size}")

    model = create_model_from_config(config).to(device)
    if pretrained_model_path is None:
        pretrained_model_path = find_latest_checkpoint(checkpoint_dir, PRETRAIN_STAGE)
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        print(f"Loading pretrained model from {pretrained_model_path}")
        model.load_state_dict(torch.load(pretrained_model_path, map_location=device), strict=True)
    else:
        print("Warning: Pretrained model not found. Training from scratch.")

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    cos_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        # writer = SummaryWriter(f'runs/sft_{epoch}_loss')
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)  # [B,L,vocab_size]
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))

            # 检查 loss 是否为 NaN
            if torch.isnan(loss):
                print(f"\nWarning: NaN loss detected at epoch {epoch+1}, batch_idx {batch_idx}. Skipping update.")
                print(f"Input shape: {inputs.shape}")
                print(f"Input sample: {inputs[0, :20]}") 
                continue 

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            # writer.add_scalar('Loss/train', loss.item(), batch_idx)
            pbar.set_postfix({'loss': f'{loss.item():.4f}','total_loss': f'{total_loss:.4f}'})

        if (epoch + 1) % save_every == 0:
            torch.save(model.state_dict(), build_checkpoint_path(checkpoint_dir, SFT_STAGE, epoch + 1))
        cos_scheduler.step()
        avg_loss = total_loss / len(train_loader)
        t_writer.add_scalar('Loss/train', avg_loss, epoch)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        # writer.close()

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument("--sft",action="store_true")
    parser.add_argument("--config", default=None)
    parser.add_argument("--train-data-path", default=None)
    parser.add_argument("--sft-data-path", default=None)
    parser.add_argument("--ckpt-dir", default=None)
    parser.add_argument("--pretrained-model-path", default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config_path = args.config or get_default_config_path(current_dir, sft=args.sft)
    config = load_experiment_config(config_path, current_dir)
    training_config = config["training"]
    checkpoint_config = config["checkpoint"]
    logging_config = config.get("logging", {})
    device = args.device or training_config.get("device") or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(args)

    if not args.sft:
        data_path = args.train_data_path or config["data"]["train_path"]
        if os.path.exists(data_path):
            print("strat training...")
            tensorboard_dir = logging_config.get("tensorboard_dir", os.path.join(current_dir, "runs", "total_loss"))
            t_writer = SummaryWriter(tensorboard_dir)
            train_model(
                config,
                data_path,
                training_config["learning_rate"],
                training_config["num_epochs"],
                device,
                t_writer,
                args.ckpt_dir if args.ckpt_dir is not None else checkpoint_config["dir"],
                args.save_every if args.save_every is not None else training_config.get("save_every", 1),
            )
            t_writer.close()
        else:
            print(f"Error: training data not found at {data_path}")
    
    elif args.sft:
        data_path = args.sft_data_path or config["data"]["sft_path"]
        if os.path.exists(data_path):
            print("strat training...")
            tensorboard_dir = logging_config.get("tensorboard_dir", os.path.join(current_dir, "runs", "sft", "sft_total_loss"))
            t_writer = SummaryWriter(tensorboard_dir)
            sft_train(
                config,
                data_path,
                training_config["learning_rate"],
                training_config["num_epochs"],
                device,
                t_writer,
                args.ckpt_dir if args.ckpt_dir is not None else checkpoint_config["dir"],
                args.pretrained_model_path if args.pretrained_model_path is not None else checkpoint_config.get("pretrained_model_path"),
                args.save_every if args.save_every is not None else training_config.get("save_every", 10),
            )
            t_writer.close()
        else:
            print(f"Error: SFT data not found at {data_path}")
