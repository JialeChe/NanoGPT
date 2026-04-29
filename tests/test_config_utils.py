import os
import tempfile
import unittest

from config_utils import create_model_from_config, get_block_size, load_experiment_config


class ConfigUtilsTest(unittest.TestCase):
    def test_load_experiment_config_resolves_relative_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.yaml")
            with open(config_path, "w", encoding="utf-8") as handle:
                handle.write(
                    "model:\n"
                    "  vocab_size: 32\n"
                    "  d_model: 16\n"
                    "  n_layer: 2\n"
                    "  n_head: 4\n"
                    "  d_hidden: 32\n"
                    "  max_seq_len: 12\n"
                    "data:\n"
                    "  train_path: dataset/train.txt\n"
                    "  block_size: 12\n"
                    "checkpoint:\n"
                    "  dir: ckpt\n"
                    "logging:\n"
                    "  tensorboard_dir: runs/test\n"
                    "inference:\n"
                    "  model_path: ckpt/sft/sft_epoch_1.pth\n"
                )

            config = load_experiment_config(config_path, tmpdir)

            self.assertEqual(config["data"]["train_path"], os.path.join(tmpdir, "dataset/train.txt"))
            self.assertEqual(config["checkpoint"]["dir"], os.path.join(tmpdir, "ckpt"))
            self.assertEqual(config["logging"]["tensorboard_dir"], os.path.join(tmpdir, "runs/test"))
            self.assertEqual(config["inference"]["model_path"], os.path.join(tmpdir, "ckpt/sft/sft_epoch_1.pth"))

    def test_create_model_from_config_uses_yaml_fields(self):
        config = {
            "model": {
                "vocab_size": 32,
                "d_model": 16,
                "n_layer": 2,
                "n_head": 4,
                "d_hidden": 32,
                "dropout": 0.0,
                "use_moe": False,
                "max_seq_len": 12,
            },
            "data": {"block_size": 12},
        }

        model = create_model_from_config(config)

        self.assertEqual(model.max_seq_len, 12)
        self.assertEqual(model.token_emb.num_embeddings, 32)
        self.assertEqual(len(model.layers), 2)
        self.assertEqual(get_block_size(config), 12)


if __name__ == "__main__":
    unittest.main()