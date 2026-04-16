"""Tests for upload_model_hf module.

All HuggingFace Hub dependencies are mocked so these tests run without network
access or authentication.
"""

import argparse
import pytest
from unittest.mock import MagicMock, patch

from src.llm_finetune.upload_model_hf import (
    build_repo_id,
    detect_upload_type,
    upload,
)


# ---------------------------------------------------------------------------
# detect_upload_type
# ---------------------------------------------------------------------------


class TestDetectUploadType:
    """Tests for the detect_upload_type heuristic."""

    def test_single_gguf_file(self, tmp_path):
        gguf_file = tmp_path / "model-q4_k_m.gguf"
        gguf_file.write_bytes(b"GGUF")
        assert detect_upload_type(gguf_file) == "gguf"

    def test_directory_with_gguf_files(self, tmp_path):
        (tmp_path / "model-q4.gguf").write_bytes(b"GGUF")
        (tmp_path / "README.md").write_text("hello")
        assert detect_upload_type(tmp_path) == "gguf"

    def test_lora_adapter(self, tmp_path):
        (tmp_path / "adapter_config.json").write_text("{}")
        (tmp_path / "adapter_model.safetensors").write_bytes(b"")
        assert detect_upload_type(tmp_path) == "lora"

    def test_checkpoint(self, tmp_path):
        (tmp_path / "optimizer.pt").write_bytes(b"")
        (tmp_path / "scheduler.pt").write_bytes(b"")
        (tmp_path / "trainer_state.json").write_text("{}")
        (tmp_path / "training_args.bin").write_bytes(b"")
        assert detect_upload_type(tmp_path) == "checkpoint"

    def test_checkpoint_partial_markers(self, tmp_path):
        (tmp_path / "trainer_state.json").write_text("{}")
        (tmp_path / "model.safetensors").write_bytes(b"")
        assert detect_upload_type(tmp_path) == "checkpoint"

    def test_merged_model_safetensors(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "model.safetensors").write_bytes(b"")
        assert detect_upload_type(tmp_path) == "merged"

    def test_merged_model_bin(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "pytorch_model.bin").write_bytes(b"")
        assert detect_upload_type(tmp_path) == "merged"

    def test_unknown(self, tmp_path):
        (tmp_path / "random_file.txt").write_text("hello")
        assert detect_upload_type(tmp_path) == "unknown"

    def test_gguf_takes_priority_over_other_markers(self, tmp_path):
        """GGUF detection runs before other checks."""
        (tmp_path / "model.gguf").write_bytes(b"GGUF")
        (tmp_path / "adapter_config.json").write_text("{}")
        assert detect_upload_type(tmp_path) == "gguf"


# ---------------------------------------------------------------------------
# build_repo_id
# ---------------------------------------------------------------------------


class TestBuildRepoId:
    def test_with_namespace(self):
        args = argparse.Namespace(
            namespace="my-org", username="alice", model_name="my-model"
        )
        assert build_repo_id(args) == "my-org/my-model"

    def test_falls_back_to_username(self):
        args = argparse.Namespace(
            namespace=None, username="alice", model_name="my-model"
        )
        assert build_repo_id(args) == "alice/my-model"


# ---------------------------------------------------------------------------
# check_auth
# ---------------------------------------------------------------------------


class TestCheckAuth:
    @patch("src.llm_finetune.upload_model_hf.whoami", create=True)
    def test_success(self, mock_whoami):
        with patch.dict(
            "sys.modules",
            {"huggingface_hub": MagicMock(whoami=lambda: {"name": "bob"})},
        ):
            # Re-import to pick up mocked module
            from importlib import reload
            import src.llm_finetune.upload_model_hf as mod

            reload(mod)
            assert mod.check_auth() == "bob"

    def test_failure_exits(self):
        with patch.dict(
            "sys.modules",
            {
                "huggingface_hub": MagicMock(
                    whoami=MagicMock(side_effect=Exception("no token"))
                ),
            },
        ):
            from importlib import reload
            import src.llm_finetune.upload_model_hf as mod

            reload(mod)
            with pytest.raises(SystemExit):
                mod.check_auth()


# ---------------------------------------------------------------------------
# upload
# ---------------------------------------------------------------------------


def _make_upload_args(model_path, **overrides):
    """Create a minimal args namespace for upload()."""
    defaults = dict(
        model_path=str(model_path),
        model_name="test-model",
        namespace=None,
        username="testuser",
        type=None,
        public=False,
        commit_message="test upload",
        include_optimizer_states=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestUpload:
    @patch("src.llm_finetune.upload_model_hf.HfApi", create=True)
    def test_upload_merged_folder(self, mock_hfapi_cls, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "model.safetensors").write_bytes(b"")

        mock_api = MagicMock()
        mock_hfapi_cls.return_value = mock_api

        with patch.dict(
            "sys.modules",
            {"huggingface_hub": MagicMock(HfApi=mock_hfapi_cls)},
        ):
            from importlib import reload
            import src.llm_finetune.upload_model_hf as mod

            reload(mod)

            args = _make_upload_args(tmp_path)
            mod.upload(args)

        mock_api.upload_folder.assert_called_once()
        call_kwargs = mock_api.upload_folder.call_args[1]
        assert call_kwargs["repo_id"] == "testuser/test-model"
        assert call_kwargs["ignore_patterns"] is None

    @patch("src.llm_finetune.upload_model_hf.HfApi", create=True)
    def test_upload_single_gguf_file(self, mock_hfapi_cls, tmp_path):
        gguf_file = tmp_path / "model-q4.gguf"
        gguf_file.write_bytes(b"GGUF")

        mock_api = MagicMock()
        mock_hfapi_cls.return_value = mock_api

        with patch.dict(
            "sys.modules",
            {"huggingface_hub": MagicMock(HfApi=mock_hfapi_cls)},
        ):
            from importlib import reload
            import src.llm_finetune.upload_model_hf as mod

            reload(mod)

            args = _make_upload_args(gguf_file)
            mod.upload(args)

        mock_api.upload_file.assert_called_once()
        call_kwargs = mock_api.upload_file.call_args[1]
        assert call_kwargs["path_in_repo"] == "model-q4.gguf"
        assert call_kwargs["repo_id"] == "testuser/test-model"

    @patch("src.llm_finetune.upload_model_hf.HfApi", create=True)
    def test_upload_gguf_directory(self, mock_hfapi_cls, tmp_path):
        (tmp_path / "model-q4.gguf").write_bytes(b"GGUF")
        (tmp_path / "model-q8.gguf").write_bytes(b"GGUF")

        mock_api = MagicMock()
        mock_hfapi_cls.return_value = mock_api

        with patch.dict(
            "sys.modules",
            {"huggingface_hub": MagicMock(HfApi=mock_hfapi_cls)},
        ):
            from importlib import reload
            import src.llm_finetune.upload_model_hf as mod

            reload(mod)

            args = _make_upload_args(tmp_path)
            mod.upload(args)

        mock_api.upload_folder.assert_called_once()

    @patch("src.llm_finetune.upload_model_hf.HfApi", create=True)
    def test_upload_checkpoint_ignores_optimizer(
        self, mock_hfapi_cls, tmp_path
    ):
        (tmp_path / "optimizer.pt").write_bytes(b"")
        (tmp_path / "trainer_state.json").write_text("{}")

        mock_api = MagicMock()
        mock_hfapi_cls.return_value = mock_api

        with patch.dict(
            "sys.modules",
            {"huggingface_hub": MagicMock(HfApi=mock_hfapi_cls)},
        ):
            from importlib import reload
            import src.llm_finetune.upload_model_hf as mod

            reload(mod)

            args = _make_upload_args(tmp_path)
            mod.upload(args)

        call_kwargs = mock_api.upload_folder.call_args[1]
        assert call_kwargs["ignore_patterns"] is not None
        assert "optimizer.pt" in call_kwargs["ignore_patterns"]

    @patch("src.llm_finetune.upload_model_hf.HfApi", create=True)
    def test_upload_checkpoint_keeps_optimizer_when_requested(
        self, mock_hfapi_cls, tmp_path
    ):
        (tmp_path / "optimizer.pt").write_bytes(b"")
        (tmp_path / "trainer_state.json").write_text("{}")

        mock_api = MagicMock()
        mock_hfapi_cls.return_value = mock_api

        with patch.dict(
            "sys.modules",
            {"huggingface_hub": MagicMock(HfApi=mock_hfapi_cls)},
        ):
            from importlib import reload
            import src.llm_finetune.upload_model_hf as mod

            reload(mod)

            args = _make_upload_args(tmp_path, include_optimizer_states=True)
            mod.upload(args)

        call_kwargs = mock_api.upload_folder.call_args[1]
        assert call_kwargs["ignore_patterns"] is None

    def test_upload_nonexistent_path_exits(self, tmp_path):
        args = _make_upload_args(tmp_path / "does_not_exist")
        with pytest.raises(SystemExit):
            upload(args)

    @patch("src.llm_finetune.upload_model_hf.HfApi", create=True)
    def test_upload_type_override(self, mock_hfapi_cls, tmp_path):
        """--type flag overrides auto-detection."""
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "model.safetensors").write_bytes(b"")

        mock_api = MagicMock()
        mock_hfapi_cls.return_value = mock_api

        with patch.dict(
            "sys.modules",
            {"huggingface_hub": MagicMock(HfApi=mock_hfapi_cls)},
        ):
            from importlib import reload
            import src.llm_finetune.upload_model_hf as mod

            reload(mod)

            args = _make_upload_args(tmp_path, type="lora")
            mod.upload(args)

        # Should still upload (folder), just with overridden type
        mock_api.upload_folder.assert_called_once()

    @patch("src.llm_finetune.upload_model_hf.HfApi", create=True)
    def test_upload_public_repo(self, mock_hfapi_cls, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "model.safetensors").write_bytes(b"")

        mock_api = MagicMock()
        mock_hfapi_cls.return_value = mock_api

        with patch.dict(
            "sys.modules",
            {"huggingface_hub": MagicMock(HfApi=mock_hfapi_cls)},
        ):
            from importlib import reload
            import src.llm_finetune.upload_model_hf as mod

            reload(mod)

            args = _make_upload_args(tmp_path, public=True)
            mod.upload(args)

        call_kwargs = mock_api.create_repo.call_args[1]
        assert call_kwargs["private"] is False

    @patch("src.llm_finetune.upload_model_hf.HfApi", create=True)
    def test_upload_gguf_explicit_type_on_single_file(
        self, mock_hfapi_cls, tmp_path
    ):
        """--type gguf on a single file routes through upload_file."""
        gguf_file = tmp_path / "model-q4.gguf"
        gguf_file.write_bytes(b"GGUF")

        mock_api = MagicMock()
        mock_hfapi_cls.return_value = mock_api

        with patch.dict(
            "sys.modules",
            {"huggingface_hub": MagicMock(HfApi=mock_hfapi_cls)},
        ):
            from importlib import reload
            import src.llm_finetune.upload_model_hf as mod

            reload(mod)

            args = _make_upload_args(gguf_file, type="gguf")
            mod.upload(args)

        mock_api.upload_file.assert_called_once()
        mock_api.upload_folder.assert_not_called()

    @patch("src.llm_finetune.upload_model_hf.HfApi", create=True)
    def test_upload_gguf_explicit_type_on_directory(
        self, mock_hfapi_cls, tmp_path
    ):
        """--type gguf on a directory routes through upload_folder."""
        (tmp_path / "model-q4.gguf").write_bytes(b"GGUF")
        (tmp_path / "model-q8.gguf").write_bytes(b"GGUF")

        mock_api = MagicMock()
        mock_hfapi_cls.return_value = mock_api

        with patch.dict(
            "sys.modules",
            {"huggingface_hub": MagicMock(HfApi=mock_hfapi_cls)},
        ):
            from importlib import reload
            import src.llm_finetune.upload_model_hf as mod

            reload(mod)

            args = _make_upload_args(tmp_path, type="gguf")
            mod.upload(args)

        mock_api.upload_folder.assert_called_once()
        mock_api.upload_file.assert_not_called()

    @patch("src.llm_finetune.upload_model_hf.HfApi", create=True)
    def test_upload_gguf_no_optimizer_ignore_patterns(
        self, mock_hfapi_cls, tmp_path
    ):
        """GGUF uploads do not set ignore_patterns (not a checkpoint)."""
        (tmp_path / "model-q4.gguf").write_bytes(b"GGUF")

        mock_api = MagicMock()
        mock_hfapi_cls.return_value = mock_api

        with patch.dict(
            "sys.modules",
            {"huggingface_hub": MagicMock(HfApi=mock_hfapi_cls)},
        ):
            from importlib import reload
            import src.llm_finetune.upload_model_hf as mod

            reload(mod)

            args = _make_upload_args(tmp_path)
            mod.upload(args)

        call_kwargs = mock_api.upload_folder.call_args[1]
        assert call_kwargs["ignore_patterns"] is None
