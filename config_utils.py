import os
from copy import deepcopy

import yaml

from model.decoder_transformer import decoder_transformer


PRETRAIN_CONFIG = os.path.join("configs", "pre_train.yaml")
SFT_CONFIG = os.path.join("configs", "sft.yaml")


def get_default_config_path(project_root, sft=False):
    config_name = SFT_CONFIG if sft else PRETRAIN_CONFIG
    return os.path.join(project_root, config_name)


def _resolve_path(project_root, path_value):
    if not path_value:
        return path_value
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(project_root, path_value)


def load_experiment_config(config_path, project_root):
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    resolved = deepcopy(config)
    data_config = resolved.setdefault("data", {})
    checkpoint_config = resolved.setdefault("checkpoint", {})
    logging_config = resolved.setdefault("logging", {})
    inference_config = resolved.setdefault("inference", {})

    for key in ("train_path", "sft_path"):
        if key in data_config:
            data_config[key] = _resolve_path(project_root, data_config[key])

    for key in ("dir", "pretrained_model_path"):
        if key in checkpoint_config:
            checkpoint_config[key] = _resolve_path(project_root, checkpoint_config[key])

    if "tensorboard_dir" in logging_config:
        logging_config["tensorboard_dir"] = _resolve_path(project_root, logging_config["tensorboard_dir"])

    if "model_path" in inference_config:
        inference_config["model_path"] = _resolve_path(project_root, inference_config["model_path"])

    resolved["config_path"] = os.path.abspath(config_path)
    resolved["project_root"] = project_root
    return resolved


def get_model_kwargs(config):
    model_config = dict(config["model"])
    if "d_hidden" not in model_config:
        model_config["d_hidden"] = model_config["d_model"] * 2
    return model_config


def create_model_from_config(config):
    return decoder_transformer(**get_model_kwargs(config))


def get_block_size(config):
    data_config = config.get("data", {})
    if "block_size" in data_config:
        return data_config["block_size"]
    return config["model"]["max_seq_len"]
