import json
import os
import sys
import tempfile
import types
import unittest


class _FakeEncoding:
    n_vocab = 1024
    eot_token = 999

    def encode(self, text, allowed_special=None):
        return [ord(char) % 128 for char in text]

    def decode(self, tokens):
        return "".join("<|endoftext|>" if token == self.eot_token else chr((token % 26) + 97) for token in tokens)


fake_tiktoken = types.SimpleNamespace(get_encoding=lambda name: _FakeEncoding())
sys.modules.setdefault("tiktoken", fake_tiktoken)

from dataset.dataloader import sft_dataset


class SFTDatasetTest(unittest.TestCase):
    def test_long_sample_truncation_keeps_prompt_masking(self):
        sample = [
            {
                "instruction": "a",
                "input": "b",
                "output": "c" * 64,
            }
        ]

        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as handle:
            json.dump(sample, handle, ensure_ascii=False)
            data_path = handle.name

        try:
            dataset = sft_dataset(data_path, max_len=24)
            x, y = dataset[0]
        finally:
            os.remove(data_path)

        self.assertEqual(x.shape[0], 24)
        self.assertEqual(y.shape[0], 24)
        self.assertTrue((y == -100).any().item())
        self.assertTrue((y != -100).any().item())


if __name__ == "__main__":
    unittest.main()