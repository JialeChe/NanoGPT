import os
import re
from glob import glob


PRETRAIN_STAGE = "pretrain"
SFT_STAGE = "sft"


def ensure_checkpoint_dirs(checkpoint_root):
    pretrain_dir = os.path.join(checkpoint_root, PRETRAIN_STAGE)
    sft_dir = os.path.join(checkpoint_root, SFT_STAGE)
    os.makedirs(pretrain_dir, exist_ok=True)
    os.makedirs(sft_dir, exist_ok=True)
    return {
        "root": checkpoint_root,
        PRETRAIN_STAGE: pretrain_dir,
        SFT_STAGE: sft_dir,
    }


def build_checkpoint_path(checkpoint_root, stage, epoch):
    checkpoint_dirs = ensure_checkpoint_dirs(checkpoint_root)
    if stage == PRETRAIN_STAGE:
        filename = f"model_epoch_{epoch}.pth"
    elif stage == SFT_STAGE:
        filename = f"sft_epoch_{epoch}.pth"
    else:
        raise ValueError(f"Unsupported checkpoint stage: {stage}")
    return os.path.join(checkpoint_dirs[stage], filename)


def find_latest_checkpoint(checkpoint_root, stage):
    checkpoint_dirs = ensure_checkpoint_dirs(checkpoint_root)
    if stage == PRETRAIN_STAGE:
        pattern = re.compile(r"model_epoch_(\d+)\.pth$")
    elif stage == SFT_STAGE:
        pattern = re.compile(r"sft_epoch_(\d+)\.pth$")
    else:
        raise ValueError(f"Unsupported checkpoint stage: {stage}")

    latest_path = None
    latest_epoch = -1
    for candidate in glob(os.path.join(checkpoint_dirs[stage], "*.pth")):
        match = pattern.search(os.path.basename(candidate))
        if not match:
            continue
        epoch = int(match.group(1))
        if epoch > latest_epoch:
            latest_epoch = epoch
            latest_path = candidate
    return latest_path


def resolve_model_path(model_path, checkpoint_root, prefer_sft=True):
    if model_path:
        return model_path

    search_order = [SFT_STAGE, PRETRAIN_STAGE] if prefer_sft else [PRETRAIN_STAGE, SFT_STAGE]
    for stage in search_order:
        checkpoint_path = find_latest_checkpoint(checkpoint_root, stage)
        if checkpoint_path is not None:
            return checkpoint_path

    raise FileNotFoundError(f"No checkpoint found under {checkpoint_root}")


def extract_state_dict(checkpoint):
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    unwanted_prefix = "_orig_mod."
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(unwanted_prefix):
            cleaned_state_dict[key[len(unwanted_prefix):]] = value
        else:
            cleaned_state_dict[key] = value
    return cleaned_state_dict