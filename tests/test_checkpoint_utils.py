import os
import tempfile
import unittest

from checkpoint_utils import (
    PRETRAIN_STAGE,
    SFT_STAGE,
    build_checkpoint_path,
    ensure_checkpoint_dirs,
    find_latest_checkpoint,
    resolve_model_path,
)


class CheckpointUtilsTest(unittest.TestCase):
    def test_build_and_find_latest_pretrain_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dirs = ensure_checkpoint_dirs(tmpdir)
            self.assertTrue(os.path.isdir(checkpoint_dirs[PRETRAIN_STAGE]))
            self.assertTrue(os.path.isdir(checkpoint_dirs[SFT_STAGE]))

            first_path = build_checkpoint_path(tmpdir, PRETRAIN_STAGE, 1)
            latest_path = build_checkpoint_path(tmpdir, PRETRAIN_STAGE, 3)
            open(first_path, "w", encoding="utf-8").close()
            open(latest_path, "w", encoding="utf-8").close()

            self.assertEqual(find_latest_checkpoint(tmpdir, PRETRAIN_STAGE), latest_path)

    def test_resolve_model_path_prefers_sft_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pretrain_path = build_checkpoint_path(tmpdir, PRETRAIN_STAGE, 2)
            sft_path = build_checkpoint_path(tmpdir, SFT_STAGE, 4)
            open(pretrain_path, "w", encoding="utf-8").close()
            open(sft_path, "w", encoding="utf-8").close()

            self.assertEqual(resolve_model_path(None, tmpdir, prefer_sft=True), sft_path)


if __name__ == "__main__":
    unittest.main()