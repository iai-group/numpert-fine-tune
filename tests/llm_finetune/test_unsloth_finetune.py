"""Tests for unsloth_finetune module.

All GPU/model dependencies (unsloth, transformers, trl, wandb) are mocked so
these tests run on CPU-only machines.  We do NOT exercise actual training,
tokenization, or GPU code — only the orchestration logic, helper functions, and
data-routing decisions.
"""

import sys
import types
import pytest
from collections import Counter
from unittest.mock import MagicMock, patch
from datasets import Dataset
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Stub heavy third-party packages before importing the module under test.
# These must be inserted BEFORE the first ``import unsloth_finetune``.
# ---------------------------------------------------------------------------


def _ensure_module(name, attrs=None):
    """Insert a stub module if it is not already importable."""
    if name not in sys.modules:
        mod = types.ModuleType(name)
        if attrs:
            for k, v in attrs.items():
                setattr(mod, k, v)
        sys.modules[name] = mod
    return sys.modules[name]


# --- unsloth ---
_unsloth_chat_templates = _ensure_module(
    "unsloth.chat_templates",
    {"get_chat_template": MagicMock(side_effect=lambda tok, **kw: tok)},
)
_unsloth = _ensure_module(
    "unsloth",
    {
        "FastLanguageModel": MagicMock(),
        "is_bfloat16_supported": MagicMock(return_value=False),
        "chat_templates": _unsloth_chat_templates,
    },
)


# --- trl ---
# SFTTrainer must be a real class since PerturbationAwareSFTTrainer inherits it.
class _StubSFTTrainer:
    """Minimal stand-in for trl.SFTTrainer."""

    def __init__(self, *args, **kwargs):
        self.model = kwargs.get("model")
        self.tokenizer = kwargs.get("tokenizer")
        self.train_dataset = kwargs.get("train_dataset")
        self.eval_dataset = kwargs.get("eval_dataset")
        self.args = kwargs.get("args")
        self.data_collator = kwargs.get("data_collator")

    def get_train_dataloader(self):
        return MagicMock(spec=DataLoader)


_trl = _ensure_module(
    "trl",
    {
        "SFTTrainer": _StubSFTTrainer,
        "SFTConfig": MagicMock,
    },
)

# --- wandb ---
_wandb = _ensure_module("wandb", {"init": MagicMock(), "finish": MagicMock()})


# --- transformers ---
# datasets' _dill module introspects transformers for PreTrainedTokenizerBase,
# so we need a real (empty) class for issubclass checks.
class _StubPreTrainedTokenizerBase:
    pass


class _StubTrainerCallback:
    """Minimal stand-in so ValidationCallback can inherit."""

    def on_evaluate(self, *args, **kwargs):
        pass


_transformers = _ensure_module(
    "transformers",
    {
        "DataCollatorForSeq2Seq": MagicMock,
        "TrainerCallback": _StubTrainerCallback,
        "PreTrainedTokenizerBase": _StubPreTrainedTokenizerBase,
    },
)


# ---------------------------------------------------------------------------
# Mock-data helpers
# ---------------------------------------------------------------------------


def _make_conversation(claim_id, perturb_type, label, claim_text="claim"):
    """Build a single example dict that looks like expanded pipeline data."""
    return {
        "claim_id": claim_id,
        "perturbation_type": perturb_type,
        "label": label,
        "conversations": [
            {"role": "system", "content": "You are a fact-checker."},
            {
                "role": "user",
                "content": f"Claim: {claim_text}\nEvidence: some doc",
            },
            {
                "role": "assistant",
                "content": f'{{"label": {label.lower()}}}',
            },
        ],
    }


def create_mock_dataset(
    n_claims=5,
    perturbations_per_claim=3,
    include_original=True,
):
    """Create a mock HuggingFace Dataset mirroring the fine-tuning pipeline."""
    examples = []
    perturb_types = [
        "perturb_type1",
        "perturb_type2",
        "perturb_type3_random",
    ]
    for i in range(n_claims):
        cid = f"claim_{i:04d}"
        if include_original:
            examples.append(
                _make_conversation(cid, "original_normalized", "True")
            )
        for j in range(perturbations_per_claim):
            examples.append(
                _make_conversation(
                    cid,
                    perturb_types[j % len(perturb_types)],
                    "False",
                    claim_text=f"claim {i} perturbed {j}",
                )
            )
    return Dataset.from_list(examples)


def create_mock_dataset_no_assistant(n=3):
    """Dataset where conversations lack an assistant response."""
    examples = []
    for i in range(n):
        examples.append(
            {
                "claim_id": f"bad_{i}",
                "perturbation_type": "original_normalized",
                "label": "True",
                "conversations": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": f"claim {i}"},
                ],
            }
        )
    return Dataset.from_list(examples)


@pytest.fixture
def small_train_dataset():
    return create_mock_dataset(n_claims=5, perturbations_per_claim=2)


@pytest.fixture
def small_val_dataset():
    return create_mock_dataset(n_claims=3, perturbations_per_claim=1)


# ---------------------------------------------------------------------------
# Tests for get_model_name_arg
# ---------------------------------------------------------------------------


class TestGetModelNameArg:
    """Test model name resolution."""

    def test_qwen3_8b(self):
        from src.llm_finetune.unsloth_finetune import get_model_name_arg

        assert get_model_name_arg("Qwen3-8B") == "unsloth/Qwen3-8B"

    def test_qwen3_32b(self):
        from src.llm_finetune.unsloth_finetune import get_model_name_arg

        assert get_model_name_arg("Qwen3-32B") == "unsloth/Qwen3-32B"

    def test_llama_3_2_3b(self):
        from src.llm_finetune.unsloth_finetune import get_model_name_arg

        assert (
            get_model_name_arg("llama-3.2-3b")
            == "unsloth/Llama-3.2-3B-Instruct"
        )

    def test_llama_3_2_1b(self):
        from src.llm_finetune.unsloth_finetune import get_model_name_arg

        assert (
            get_model_name_arg("llama-3.2-1b")
            == "unsloth/Llama-3.2-1B-Instruct"
        )

    def test_custom_model_passthrough(self):
        from src.llm_finetune.unsloth_finetune import get_model_name_arg

        assert (
            get_model_name_arg("my-org/my-custom-model")
            == "my-org/my-custom-model"
        )

    def test_qwen3_8b_variant(self):
        from src.llm_finetune.unsloth_finetune import get_model_name_arg

        assert get_model_name_arg("Qwen3-8B-bnb-4bit") == "unsloth/Qwen3-8B"


# ---------------------------------------------------------------------------
# Tests for resolve_step_interval
# ---------------------------------------------------------------------------


class TestResolveStepInterval:
    """Tests for fractional, absolute-int, and edge-case inputs."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from src.llm_finetune.unsloth_finetune import resolve_step_interval

        self.fn = resolve_step_interval

    # --- fractional inputs (0 < value <= 1.0) ---

    def test_fraction_returns_rounded_steps(self):
        """Fraction is multiplied by total_steps and ceiled to nearest int."""
        assert self.fn(0.25, total_steps=100) == 25

    def test_fraction_rounds_to_nearest(self):
        """Ceiling is used, so 0.33 * 100 = 33.0 → 33."""
        assert self.fn(0.33, total_steps=100) == 33

    def test_fraction_ceiling_on_tie(self):
        """Ties are ceiled rather than rounded-to-even (0.5 * 5 = 2.5 → 3)."""
        assert self.fn(0.5, total_steps=5) == 3

    def test_fraction_minimum_one(self):
        """Result is clamped to at least 1 when fraction * total_steps < 1."""
        assert self.fn(0.01, total_steps=5) == 1

    def test_fraction_exactly_one(self):
        """A value of 1.0 is treated as a fraction equal to one full epoch."""
        assert self.fn(1.0, total_steps=200) == 200

    def test_fraction_requires_total_steps(self):
        """Passing total_steps=None with a fraction raises ValueError."""
        with pytest.raises(ValueError, match="total_steps is required"):
            self.fn(0.25, total_steps=None)

    def test_fraction_requires_total_steps_default(self):
        """Omitting total_steps (defaults to None)"""
        with pytest.raises(ValueError):
            self.fn(0.5)

    # --- float >= 1 (treated as absolute, cast to int) ---

    def test_float_above_one_cast_to_int(self):
        """A float >= 1 is treated as an absolute step count and cast to int."""
        assert self.fn(100.0) == 100

    def test_float_above_one_truncates(self):
        """Fractional part is discarded when casting a float >= 1 to int."""
        assert self.fn(99.9) == 99

    def test_float_zero_raises(self):
        """A float value of 0.0 raises ValueError (non-positive)."""
        with pytest.raises(ValueError, match="must be positive"):
            self.fn(0.0)

    def test_float_negative_raises(self):
        """A negative float raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            self.fn(-5.0)

    # --- integer inputs ---

    def test_integer_passthrough(self):
        """A positive integer is returned unchanged."""
        assert self.fn(500) == 500

    def test_integer_one(self):
        """The minimum valid integer value (1) is returned unchanged."""
        assert self.fn(1) == 1

    def test_integer_zero_raises(self):
        """Zero raises ValueError (step count must be positive)."""
        with pytest.raises(ValueError, match="must be positive"):
            self.fn(0)

    def test_integer_negative_raises(self):
        """A negative integer raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            self.fn(-1)

    # --- step_name appears in error messages ---

    def test_error_message_includes_step_name(self):
        """The step_name parameter is included in the ValueError message."""
        with pytest.raises(ValueError, match="eval_steps"):
            self.fn(0.5, total_steps=None, step_name="eval_steps")

    def test_error_message_includes_value(self):
        """The offending value is included in the ValueError message."""
        with pytest.raises(ValueError, match="-3"):
            self.fn(-3, step_name="save_steps")


# ---------------------------------------------------------------------------
# Tests for ValidationCallback
# ---------------------------------------------------------------------------


class TestValidationCallback:
    """Test the ValidationCallback prints metrics."""

    def test_on_evaluate_prints_metrics(self, capsys):
        from src.llm_finetune.unsloth_finetune import ValidationCallback

        cb = ValidationCallback()
        metrics = {"eval_loss": 0.1234, "eval_accuracy": 0.9500}
        cb.on_evaluate(args=None, state=None, control=None, metrics=metrics)

        captured = capsys.readouterr()
        assert "VALIDATION METRICS" in captured.out
        assert "eval_loss" in captured.out
        assert "0.1234" in captured.out
        assert "0.9500" in captured.out

    def test_on_evaluate_no_metrics(self, capsys):
        from src.llm_finetune.unsloth_finetune import ValidationCallback

        cb = ValidationCallback()
        cb.on_evaluate(args=None, state=None, control=None, metrics=None)

        captured = capsys.readouterr()
        assert "VALIDATION METRICS" not in captured.out


# ---------------------------------------------------------------------------
# Tests for PerturbationAwareSFTTrainer
# ---------------------------------------------------------------------------


class TestPerturbationAwareSFTTrainer:
    """Test the custom trainer's dataloader routing logic."""

    def _make_trainer(self, use_custom_sampler, batch_sampler=None):
        """Build a PerturbationAwareSFTTrainer with minimal init."""
        from src.llm_finetune.unsloth_finetune import (
            PerturbationAwareSFTTrainer,
        )

        # We bypass _StubSFTTrainer.__init__ and set only the attrs that
        # get_train_dataloader inspects.
        trainer = object.__new__(PerturbationAwareSFTTrainer)
        trainer.use_custom_sampler = use_custom_sampler
        trainer.pre_built_batch_sampler = batch_sampler
        trainer.train_dataset = create_mock_dataset(n_claims=3)
        trainer.data_collator = lambda x: x

        # Mock the args object that DataLoader needs
        trainer.args = MagicMock()
        trainer.args.dataloader_num_workers = 0
        trainer.args.dataloader_pin_memory = False

        # Mock the _train_batch_size attribute needed by parent's
        # get_train_dataloader
        trainer._train_batch_size = 8

        # Ensure _signature_columns exists (needed when calling parent methods)
        trainer._signature_columns = None

        return trainer

    def test_defaults_to_super_when_sampler_disabled(self):
        """When custom sampler is off, falls back to parent dataloader."""
        trainer = self._make_trainer(use_custom_sampler=False)
        dl = trainer.get_train_dataloader()
        # The parent _StubSFTTrainer.get_train_dataloader returns a MagicMock
        assert dl is not None

    def test_defaults_to_super_when_no_batch_sampler(self):
        """Custom sampler on but pre_built_batch_sampler is None -> parent."""
        trainer = self._make_trainer(
            use_custom_sampler=True, batch_sampler=None
        )
        dl = trainer.get_train_dataloader()
        assert dl is not None

    def test_custom_dataloader_with_batch_sampler(self):
        """When custom sampler is on and provided, returns a real DataLoader."""
        mock_sampler = MagicMock()
        mock_sampler.__len__ = MagicMock(return_value=5)
        mock_sampler.__iter__ = MagicMock(
            return_value=iter([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
        )

        trainer = self._make_trainer(
            use_custom_sampler=True, batch_sampler=mock_sampler
        )
        dl = trainer.get_train_dataloader()
        assert isinstance(dl, DataLoader)

    def test_raises_when_no_train_dataset(self):
        """Should raise if train_dataset is None."""
        mock_sampler = MagicMock()
        mock_sampler.__len__ = MagicMock(return_value=5)

        trainer = self._make_trainer(
            use_custom_sampler=True, batch_sampler=mock_sampler
        )
        trainer.train_dataset = None

        with pytest.raises(ValueError, match="training requires a train"):
            trainer.get_train_dataloader()


# ---------------------------------------------------------------------------
# Tests for finetune() orchestration
#
# The function is large and touches GPU/model code, so we mock everything
# past the point of data loading.  The goal is to verify the *routing*
# logic: which objects get created, in what order, and with which args.
# ---------------------------------------------------------------------------


def _make_mock_trainer():
    """Create a mock trainer instance returned by SFTTrainer(...)."""
    trainer = MagicMock()
    trainer.state.epoch = 1.0
    trainer.evaluate.return_value = {"eval_loss": 0.25, "eval_runtime": 1.0}
    return trainer


def _tokenized_dataset(n=6):
    """Return a dataset already in tensor-column shape (post-tokenize).

    This is what the data looks like *after* tokenize_func + set_format.
    """
    return Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3, 4, 5]] * n,
            "attention_mask": [[1, 1, 1, 1, 1]] * n,
            "labels": [[-100, -100, 3, 4, 5]] * n,
        }
    )


class TestFinetune:
    """Test finetune() orchestration with all heavy deps mocked.

    We patch the entire function body so it never actually tokenizes, trains, or
    touches a GPU.  We only verify that the right branches are taken and helpers
    are called with the expected arguments.
    """

    def _finetune_context(self, tmp_path, **extra_patches):
        """Return a context manager that patches all heavy deps."""
        import contextlib

        mock_model = MagicMock()
        mock_tok = MagicMock()
        mock_tok.padding_side = "right"
        mock_tok.pad_token = None
        mock_tok.eos_token = "<eos>"
        mock_tok.apply_chat_template = MagicMock(
            side_effect=lambda conv, **kw: " ".join(m["content"] for m in conv)
        )
        mock_tok.return_value = {
            "input_ids": [1, 2, 3, 4, 5],
            "attention_mask": [1, 1, 1, 1, 1],
        }

        mock_flm = MagicMock()
        mock_flm.from_pretrained.return_value = (mock_model, mock_tok)
        mock_flm.get_peft_model.return_value = mock_model

        mock_trainer = _make_mock_trainer()

        train_ds = create_mock_dataset(4, 2)
        val_ds = create_mock_dataset(2, 1)

        patches = {
            "FastLanguageModel": mock_flm,
            "get_chat_template": MagicMock(side_effect=lambda tok, **kw: tok),
            "is_bfloat16_supported": MagicMock(return_value=False),
            "load_split": MagicMock(
                side_effect=lambda **kw: train_ds
                if kw.get("split") == "train"
                else val_ds
            ),
            "SFTTrainer": MagicMock(return_value=mock_trainer),
            "SFTConfig": MagicMock(return_value=MagicMock()),
            "DataCollatorForSeq2Seq": MagicMock(return_value=MagicMock()),
            "wandb": MagicMock(),
            "prefix": str(tmp_path) + "/",
            "build_training_config": MagicMock(return_value={}),
            "save_training_config": MagicMock(
                return_value=str(tmp_path / "config.json")
            ),
        }
        patches.update(extra_patches)

        prefix = "src.llm_finetune.unsloth_finetune."

        @contextlib.contextmanager
        def ctx():
            stack = contextlib.ExitStack()
            refs = {}
            with stack:
                for name, value in patches.items():
                    p = stack.enter_context(patch(prefix + name, value))
                    refs[name] = p
                # Make wandb.init return a mock run
                refs["wandb"].init.return_value = MagicMock()
                yield {
                    "model": mock_model,
                    "tokenizer": mock_tok,
                    "flm": mock_flm,
                    "trainer": mock_trainer,
                    "refs": refs,
                }

        return ctx()

    # -- Smoke test --

    def test_basic_finetune_runs(self, tmp_path):
        """Finetune() completes without error."""
        from src.llm_finetune.unsloth_finetune import finetune

        with self._finetune_context(tmp_path) as m:
            finetune(
                train_json_paths=["t.json"],
                val_json_paths=["v.json"],
                model_name="test-model",
                perturb_type_names="test",
                max_seq_length=512,
            )

            m["flm"].from_pretrained.assert_called_once()
            m["flm"].get_peft_model.assert_called_once()
            m["trainer"].train.assert_called_once()
            m["trainer"].evaluate.assert_called_once()
            m["model"].save_pretrained.assert_called_once()
            m["model"].save_pretrained_merged.assert_called_once()

    # -- wandb --

    def test_wandb_initialized(self, tmp_path):
        """wandb.init is called with the correct project name."""
        from src.llm_finetune.unsloth_finetune import finetune

        with self._finetune_context(tmp_path) as m:
            finetune(
                train_json_paths=["t.json"],
                val_json_paths=["v.json"],
                model_name="test-model",
                perturb_type_names="test",
                max_seq_length=512,
            )
            m["refs"]["wandb"].init.assert_called_once()
            call_kwargs = m["refs"]["wandb"].init.call_args[1]
            assert call_kwargs["project"] == "numpert_finetune"

    # -- Resume logic --

    def test_resume_requires_model_path(self, tmp_path):
        """start_epoch > 0 without resume_model should raise."""
        from src.llm_finetune.unsloth_finetune import finetune

        with self._finetune_context(tmp_path):
            with pytest.raises(ValueError, match="--resume_model is required"):
                finetune(
                    train_json_paths=["t.json"],
                    val_json_paths=["v.json"],
                    model_name="test",
                    start_epoch=2,
                    resume_model=None,
                )

    def test_resume_uses_provided_model_path(self, tmp_path):
        """Resumed training loads the model from resume_model path."""
        from src.llm_finetune.unsloth_finetune import finetune

        with self._finetune_context(tmp_path) as m:
            finetune(
                train_json_paths=["t.json"],
                val_json_paths=["v.json"],
                model_name="test",
                perturb_type_names="test",
                start_epoch=2,
                resume_model="/path/to/resumed_model",
                max_seq_length=512,
            )
            load_call = m["flm"].from_pretrained.call_args
            assert load_call[1]["model_name"] == "/path/to/resumed_model"

    # -- GGUF --

    def test_gguf_saved_when_flag_set(self, tmp_path):
        """save_gguf=True triggers save_as_gguf."""
        from src.llm_finetune.unsloth_finetune import finetune

        mock_gguf = MagicMock()
        with self._finetune_context(tmp_path, save_as_gguf=mock_gguf):
            finetune(
                train_json_paths=["t.json"],
                val_json_paths=["v.json"],
                model_name="test",
                perturb_type_names="test",
                max_seq_length=512,
                save_gguf=True,
                gguf_quantization_method="q4_k_m",
            )
            mock_gguf.assert_called_once()
            assert "q4_k_m" in str(mock_gguf.call_args)

    def test_gguf_not_saved_when_flag_off(self, tmp_path):
        """save_gguf=False (default) does not call save_as_gguf."""
        from src.llm_finetune.unsloth_finetune import finetune

        mock_gguf = MagicMock()
        with self._finetune_context(tmp_path, save_as_gguf=mock_gguf):
            finetune(
                train_json_paths=["t.json"],
                val_json_paths=["v.json"],
                model_name="test",
                perturb_type_names="test",
                max_seq_length=512,
                save_gguf=False,
            )
            mock_gguf.assert_not_called()

    # -- Batch match claim_id --

    def test_batch_match_creates_custom_trainer(self, tmp_path):
        """batch_match_claim_id=True uses PerturbationAwareSFTTrainer."""
        from src.llm_finetune.unsloth_finetune import finetune

        mock_custom_cls = MagicMock(return_value=_make_mock_trainer())
        with self._finetune_context(
            tmp_path,
            PerturbationAwareSFTTrainer=mock_custom_cls,
            EnhancedPerturbationBatchSampler=MagicMock(),
        ):
            finetune(
                train_json_paths=["t.json"],
                val_json_paths=["v.json"],
                model_name="test",
                perturb_type_names="test",
                max_seq_length=512,
                batch_match_claim_id=True,
                batch_size=4,
            )
            mock_custom_cls.assert_called_once()
            call_kwargs = mock_custom_cls.call_args[1]
            assert call_kwargs["use_custom_sampler"] is True
            assert call_kwargs["pre_built_batch_sampler"] is not None

    def test_no_batch_match_uses_standard_trainer(self, tmp_path):
        """batch_match_claim_id=False uses standard SFTTrainer."""
        from src.llm_finetune.unsloth_finetune import finetune

        with self._finetune_context(tmp_path) as m:
            finetune(
                train_json_paths=["t.json"],
                val_json_paths=["v.json"],
                model_name="test",
                perturb_type_names="test",
                max_seq_length=512,
                batch_match_claim_id=False,
            )
            # Standard SFTTrainer should have been called (our mock)
            m["refs"]["SFTTrainer"].assert_called_once()

    # -- Chat template selection --

    def test_chat_template_llama(self, tmp_path):
        """Llama models get llama-3.1 chat template."""
        from src.llm_finetune.unsloth_finetune import finetune

        mock_template = MagicMock(side_effect=lambda tok, **kw: tok)
        with self._finetune_context(tmp_path, get_chat_template=mock_template):
            finetune(
                train_json_paths=["t.json"],
                val_json_paths=["v.json"],
                model_name="llama-3.2-3b",
                perturb_type_names="test",
                max_seq_length=512,
            )
            mock_template.assert_called_once()
            assert mock_template.call_args[1]["chat_template"] == "llama-3.1"

    def test_chat_template_qwen(self, tmp_path):
        """Qwen models get chatml chat template."""
        from src.llm_finetune.unsloth_finetune import finetune

        mock_template = MagicMock(side_effect=lambda tok, **kw: tok)
        with self._finetune_context(tmp_path, get_chat_template=mock_template):
            finetune(
                train_json_paths=["t.json"],
                val_json_paths=["v.json"],
                model_name="qwen-7b-chat",
                perturb_type_names="test",
                max_seq_length=512,
            )
            mock_template.assert_called_once()
            assert mock_template.call_args[1]["chat_template"] == "chatml"

    # -- Output directories --

    def test_output_directories_created(self, tmp_path):
        """Finetune creates the expected directory structure."""
        from src.llm_finetune.unsloth_finetune import finetune

        with self._finetune_context(tmp_path):
            finetune(
                train_json_paths=["t.json"],
                val_json_paths=["v.json"],
                model_name="test-model",
                perturb_type_names="test",
                max_seq_length=512,
            )
        assert (tmp_path / "models").exists()

    # -- New dataset balancer --

    def test_balancer_called_when_enabled(self, tmp_path):
        """balance_labels=True is forwarded to load_split."""
        from src.llm_finetune.unsloth_finetune import finetune

        train_ds = create_mock_dataset(4, 2)
        val_ds = create_mock_dataset(2, 1)
        mock_load_split = MagicMock(
            side_effect=lambda **kw: train_ds
            if kw.get("split") == "train"
            else val_ds
        )

        with self._finetune_context(tmp_path, load_split=mock_load_split):
            finetune(
                train_json_paths=["t.json"],
                val_json_paths=["v.json"],
                model_name="test",
                perturb_type_names="test",
                max_seq_length=512,
                balance_labels=True,
            )
            assert mock_load_split.call_count == 2
            train_call_kwargs = mock_load_split.call_args_list[0][1]
            val_call_kwargs = mock_load_split.call_args_list[1][1]
            assert train_call_kwargs["balance_labels"] is True
            assert val_call_kwargs["split"] == "validation"
            assert val_call_kwargs["balance_labels"] is False
            assert val_call_kwargs["perturb_ratio"] is None
            assert val_call_kwargs["false_perturb_t_to_f_ratio"] is None
            assert (
                "method_balance_mode" not in val_call_kwargs
                or val_call_kwargs["method_balance_mode"] is None
            )

    def test_balancer_not_called_when_disabled(self, tmp_path):
        """balance_labels=False is forwarded to load_split."""
        from src.llm_finetune.unsloth_finetune import finetune

        train_ds = create_mock_dataset(4, 2)
        val_ds = create_mock_dataset(2, 1)
        mock_load_split = MagicMock(
            side_effect=lambda **kw: train_ds
            if kw.get("split") == "train"
            else val_ds
        )
        with self._finetune_context(tmp_path, load_split=mock_load_split):
            finetune(
                train_json_paths=["t.json"],
                val_json_paths=["v.json"],
                model_name="test",
                perturb_type_names="test",
                max_seq_length=512,
                balance_labels=False,
                perturb_ratio=1.0,
            )
            assert mock_load_split.call_count == 2
            train_call_kwargs = mock_load_split.call_args_list[0][1]
            assert train_call_kwargs["balance_labels"] is False

    def test_save_steps_defaults_to_eval_steps(self, tmp_path):
        """Unset save_steps follows the resolved eval_steps cadence."""
        from src.llm_finetune.unsloth_finetune import finetune

        with self._finetune_context(tmp_path) as m:
            finetune(
                train_json_paths=["t.json"],
                val_json_paths=["v.json"],
                model_name="test",
                perturb_type_names="test",
                max_seq_length=512,
                eval_steps=7,
                save_steps=None,
            )

            sft_config_kwargs = m["refs"]["SFTConfig"].call_args[1]
            assert sft_config_kwargs["eval_steps"] == 7
            assert sft_config_kwargs["save_steps"] == 7

    def test_save_steps_preserves_custom_override(self, tmp_path):
        """Explicit save_steps should not be overwritten by eval_steps."""
        from src.llm_finetune.unsloth_finetune import finetune

        with self._finetune_context(tmp_path) as m:
            finetune(
                train_json_paths=["t.json"],
                val_json_paths=["v.json"],
                model_name="test",
                perturb_type_names="test",
                max_seq_length=512,
                eval_steps=7,
                save_steps=3,
            )

            sft_config_kwargs = m["refs"]["SFTConfig"].call_args[1]
            assert sft_config_kwargs["eval_steps"] == 7
            assert sft_config_kwargs["save_steps"] == 3


# ---------------------------------------------------------------------------
# Tests for validation data filtering
# ---------------------------------------------------------------------------


class TestValidationFiltering:
    """Test validation data filtering logic (extracted from finetune)."""

    @staticmethod
    def _has_assistant_response(example):
        conv = example["conversations"]
        return len(conv) >= 2 and conv[-1]["role"] == "assistant"

    def test_filter_removes_bad_conversations(self):
        good_ds = create_mock_dataset(n_claims=3, perturbations_per_claim=1)
        bad_ds = create_mock_dataset_no_assistant(n=3)

        combined = Dataset.from_list(
            [good_ds[i] for i in range(len(good_ds))]
            + [bad_ds[i] for i in range(len(bad_ds))]
        )
        filtered = combined.filter(self._has_assistant_response)
        assert len(filtered) == len(good_ds)

    def test_all_bad_data_produces_empty_dataset(self):
        bad_ds = create_mock_dataset_no_assistant(n=5)
        filtered = bad_ds.filter(self._has_assistant_response)
        assert len(filtered) == 0


# ---------------------------------------------------------------------------
# Tests for CLI parse_file_config
# ---------------------------------------------------------------------------


class TestCLIArgParsing:
    """Test argparse / FileLoadConfig parsing."""

    def test_parse_file_config_simple(self):
        from src.llm_finetune.utils.data_loading.perturbation_types import (
            parse_file_config,
        )

        cfg = parse_file_config("data/train.json")
        assert cfg.path == "data/train.json"
        assert cfg.perturb_types is None

    def test_parse_file_config_with_types(self):
        from src.llm_finetune.utils.data_loading.perturbation_types import (
            parse_file_config,
            PerturbationType,
        )

        cfg = parse_file_config("data/train.json:type1,type2")
        assert cfg.path == "data/train.json"
        assert cfg.perturb_types == [
            PerturbationType.PERTURB_TYPE1,
            PerturbationType.PERTURB_TYPE2,
        ]

    def test_parse_file_config_invalid_type_raises(self):
        from src.llm_finetune.utils.data_loading.perturbation_types import (
            parse_file_config,
        )

        with pytest.raises(ValueError, match="Unknown perturbation type"):
            parse_file_config("data/train.json:nonexistent_type")


# ---------------------------------------------------------------------------
# Tests for mock dataset integrity
# ---------------------------------------------------------------------------


class TestMockDatasetIntegrity:
    """Verify the mock dataset helpers produce valid data structures."""

    def test_mock_dataset_columns(self, small_train_dataset):
        expected = {"claim_id", "perturbation_type", "label", "conversations"}
        assert expected.issubset(set(small_train_dataset.column_names))

    def test_mock_dataset_size(self, small_train_dataset):
        # 5 claims * (1 original + 2 perturbations) = 15
        assert len(small_train_dataset) == 15

    def test_all_conversations_have_assistant(self, small_train_dataset):
        for i in range(len(small_train_dataset)):
            conv = small_train_dataset[i]["conversations"]
            assert conv[-1]["role"] == "assistant"

    def test_claim_id_distribution(self, small_train_dataset):
        counter = Counter(small_train_dataset["claim_id"])
        for count in counter.values():
            assert count == 3  # 1 original + 2 perturbations

    def test_no_assistant_dataset(self):
        ds = create_mock_dataset_no_assistant(3)
        for i in range(len(ds)):
            assert ds[i]["conversations"][-1]["role"] == "user"

    def test_perturbation_type_values(self, small_train_dataset):
        types = set(small_train_dataset["perturbation_type"])
        assert "original_normalized" in types
        assert len(types) > 1

    def test_label_values(self, small_train_dataset):
        labels = set(small_train_dataset["label"])
        assert labels.issubset({"True", "False"})

    def test_no_originals(self):
        """Dataset with include_original=False has no original_normalized."""
        ds = create_mock_dataset(
            n_claims=3, perturbations_per_claim=2, include_original=False
        )
        types = set(ds["perturbation_type"])
        assert "original_normalized" not in types
        # 3 claims * 2 perturbations only
        assert len(ds) == 6

    def test_single_claim(self):
        """Dataset with a single claim is valid."""
        ds = create_mock_dataset(n_claims=1, perturbations_per_claim=1)
        assert len(ds) == 2  # 1 original + 1 perturbation
        assert ds[0]["claim_id"] == ds[1]["claim_id"]

    def test_zero_perturbations_per_claim(self):
        """Claims with zero extra perturbations still produce originals."""
        ds = create_mock_dataset(n_claims=4, perturbations_per_claim=0)
        assert len(ds) == 4
        assert all(
            ds[i]["perturbation_type"] == "original_normalized"
            for i in range(len(ds))
        )


# ---------------------------------------------------------------------------
# Additional edge-case tests for get_model_name_arg
# ---------------------------------------------------------------------------


class TestGetModelNameArgEdgeCases:
    """Edge cases for model name resolution."""

    def test_empty_string_passthrough(self):
        """Empty string is not a known prefix, so it passes through."""
        from src.llm_finetune.unsloth_finetune import get_model_name_arg

        assert get_model_name_arg("") == ""

    def test_qwen3_32b_with_suffix(self):
        """Qwen3-32B with extra suffix still resolves."""
        from src.llm_finetune.unsloth_finetune import get_model_name_arg

        assert get_model_name_arg("Qwen3-32B-bnb-4bit") == "unsloth/Qwen3-32B"

    def test_llama_3_2_3b_with_extra_suffix(self):
        """Llama name with extra text still resolves via startswith."""
        from src.llm_finetune.unsloth_finetune import get_model_name_arg

        assert (
            get_model_name_arg("llama-3.2-3b-instruct-special")
            == "unsloth/Llama-3.2-3B-Instruct"
        )

    def test_llama_3_2_1b_with_extra_suffix(self):
        """Llama-1b name with extra text still resolves."""
        from src.llm_finetune.unsloth_finetune import get_model_name_arg

        assert (
            get_model_name_arg("llama-3.2-1b-finetuned")
            == "unsloth/Llama-3.2-1B-Instruct"
        )

    def test_case_sensitive_no_match(self):
        """Model names are case-sensitive; lowercase 'qwen3-8b' passes."""
        from src.llm_finetune.unsloth_finetune import get_model_name_arg

        # 'qwen3-8b' (lowercase q) doesn't start with 'Qwen3-8B'
        assert get_model_name_arg("qwen3-8b") == "qwen3-8b"

    def test_url_like_path_passthrough(self):
        """Full HuggingFace repo path passes through as-is."""
        from src.llm_finetune.unsloth_finetune import get_model_name_arg

        path = "organization/custom-llm-v2"
        assert get_model_name_arg(path) == path

    def test_local_path_passthrough(self):
        """Absolute local path passes through unchanged."""
        from src.llm_finetune.unsloth_finetune import get_model_name_arg

        path = "/home/user/models/my-model"
        assert get_model_name_arg(path) == path

    def test_full_hf_path_qwen3_8b(self):
        """Full unsloth/Qwen3-8B path resolves to 8B, not 32B.

        Regression: previously the 32B branch had
        ``startswith("unsloth/Qwen3-8B")`` instead of
        ``startswith("unsloth/Qwen3-32B")``, so this input was
        incorrectly routed to the 32B return value.
        """
        from src.llm_finetune.unsloth_finetune import get_model_name_arg

        assert get_model_name_arg("unsloth/Qwen3-8B") == "unsloth/Qwen3-8B"

    def test_full_hf_path_qwen3_32b(self):
        """Full unsloth/Qwen3-32B path resolves to 32B."""
        from src.llm_finetune.unsloth_finetune import get_model_name_arg

        assert get_model_name_arg("unsloth/Qwen3-32B") == "unsloth/Qwen3-32B"


# ---------------------------------------------------------------------------
# Additional edge-case tests for ValidationCallback
# ---------------------------------------------------------------------------


class TestValidationCallbackEdgeCases:
    """Edge cases for ValidationCallback."""

    def test_empty_metrics_dict(self, capsys):
        """Empty metrics dict is falsy, so no output is printed."""
        from src.llm_finetune.unsloth_finetune import ValidationCallback

        cb = ValidationCallback()
        cb.on_evaluate(args=None, state=None, control=None, metrics={})

        captured = capsys.readouterr()
        # Empty dict is falsy → no header printed
        assert "VALIDATION METRICS" not in captured.out

    def test_integer_metric_values(self, capsys):
        """Integer metric values format correctly with :.4f."""
        from src.llm_finetune.unsloth_finetune import ValidationCallback

        cb = ValidationCallback()
        cb.on_evaluate(
            args=None, state=None, control=None, metrics={"epoch": 3}
        )

        captured = capsys.readouterr()
        assert "epoch" in captured.out
        assert "3.0000" in captured.out

    def test_extra_kwargs_ignored(self, capsys):
        """Extra keyword arguments do not cause an error."""
        from src.llm_finetune.unsloth_finetune import ValidationCallback

        cb = ValidationCallback()
        cb.on_evaluate(
            args=None,
            state=None,
            control=None,
            metrics={"loss": 0.5},
            model="ignored",
            tokenizer="ignored",
        )
        captured = capsys.readouterr()
        assert "loss" in captured.out


# ---------------------------------------------------------------------------
# Additional edge-case tests for PerturbationAwareSFTTrainer
# ---------------------------------------------------------------------------


class TestPerturbationAwareSFTTrainerEdgeCases:
    """Edge cases for the custom trainer dataloader routing."""

    def _make_trainer(self, use_custom_sampler, batch_sampler=None):
        from src.llm_finetune.unsloth_finetune import (
            PerturbationAwareSFTTrainer,
        )

        trainer = object.__new__(PerturbationAwareSFTTrainer)
        trainer.use_custom_sampler = use_custom_sampler
        trainer.pre_built_batch_sampler = batch_sampler
        trainer.train_dataset = create_mock_dataset(n_claims=3)
        trainer.data_collator = lambda x: x
        trainer.args = MagicMock()
        trainer.args.dataloader_num_workers = 0
        trainer.args.dataloader_pin_memory = False

        # Mock the _train_batch_size attribute needed by parent's
        # get_train_dataloader
        trainer._train_batch_size = 8

        # Ensure _signature_columns exists (needed when calling parent methods)
        trainer._signature_columns = None

        return trainer

    def test_init_stores_custom_sampler_flag(self):
        """__init__ correctly stores use_custom_sampler attribute."""
        from src.llm_finetune.unsloth_finetune import (
            PerturbationAwareSFTTrainer,
        )
        from unittest.mock import patch

        trainer = object.__new__(PerturbationAwareSFTTrainer)
        # Mock parent __init__ to avoid requiring SFTTrainer args
        with patch.object(
            PerturbationAwareSFTTrainer.__bases__[0],
            "__init__",
            return_value=None,
        ):
            PerturbationAwareSFTTrainer.__init__(
                trainer, use_custom_sampler=True
            )
        assert trainer.use_custom_sampler is True

    def test_init_stores_pre_built_sampler(self):
        """__init__ correctly stores pre_built_batch_sampler attribute."""
        from src.llm_finetune.unsloth_finetune import (
            PerturbationAwareSFTTrainer,
        )
        from unittest.mock import patch

        sampler = MagicMock()
        trainer = object.__new__(PerturbationAwareSFTTrainer)
        # Mock parent __init__ to avoid requiring SFTTrainer args
        with patch.object(
            PerturbationAwareSFTTrainer.__bases__[0],
            "__init__",
            return_value=None,
        ):
            PerturbationAwareSFTTrainer.__init__(
                trainer,
                use_custom_sampler=True,
                pre_built_batch_sampler=sampler,
            )
        assert trainer.pre_built_batch_sampler is sampler

    def test_custom_dataloader_prints_info(self, capsys):
        """Custom sampler path prints diagnostic information."""
        mock_sampler = MagicMock()
        mock_sampler.__len__ = MagicMock(return_value=3)
        mock_sampler.__iter__ = MagicMock(
            return_value=iter([[0, 1], [2, 3], [4, 5]])
        )

        trainer = self._make_trainer(
            use_custom_sampler=True, batch_sampler=mock_sampler
        )
        trainer.get_train_dataloader()

        captured = capsys.readouterr()
        assert "EnhancedPerturbationBatchSampler" in captured.out
        assert "Custom dataloader created" in captured.out

    def test_both_flags_false_falls_back(self):
        """use_custom_sampler=False with a provided sampler still falls back."""
        mock_sampler = MagicMock()
        mock_sampler.__len__ = MagicMock(return_value=5)

        trainer = self._make_trainer(
            use_custom_sampler=False, batch_sampler=mock_sampler
        )
        dl = trainer.get_train_dataloader()
        # Falls back to parent _StubSFTTrainer.get_train_dataloader
        assert dl is not None


# ---------------------------------------------------------------------------
# Additional edge-case tests for finetune() orchestration
# ---------------------------------------------------------------------------


class TestFinetuneEdgeCases:
    """Edge-case tests for the finetune() orchestration function."""

    def _finetune_context(self, tmp_path, **extra_patches):
        """Reuse the same context manager from TestFinetune."""
        return TestFinetune()._finetune_context(tmp_path, **extra_patches)

    # -- perturb_ratio != 1.0 triggers balancer --

    def test_perturb_ratio_triggers_balancer(self, tmp_path):
        """perturb_ratio is forwarded to load_split."""
        from src.llm_finetune.unsloth_finetune import finetune

        train_ds = create_mock_dataset(4, 2)
        val_ds = create_mock_dataset(2, 1)
        mock_load_split = MagicMock(
            side_effect=lambda **kw: train_ds
            if kw.get("split") == "train"
            else val_ds
        )

        with self._finetune_context(tmp_path, load_split=mock_load_split):
            finetune(
                train_json_paths=["t.json"],
                val_json_paths=["v.json"],
                model_name="test",
                perturb_type_names="test",
                max_seq_length=512,
                balance_labels=False,
                perturb_ratio=0.5,
            )
            assert mock_load_split.call_count == 2
            train_call_kwargs = mock_load_split.call_args_list[0][1]
            assert train_call_kwargs["perturb_ratio"] == 0.5

    # -- Default chat template (non-llama, non-qwen) --

    def test_default_chat_template_fallback(self, tmp_path):
        """Non-llama, non-qwen model falls back to chatml template."""
        from src.llm_finetune.unsloth_finetune import finetune

        mock_template = MagicMock(side_effect=lambda tok, **kw: tok)
        with self._finetune_context(tmp_path, get_chat_template=mock_template):
            finetune(
                train_json_paths=["t.json"],
                val_json_paths=["v.json"],
                model_name="mistral-7b-instruct",
                perturb_type_names="test",
                max_seq_length=512,
            )
            mock_template.assert_called_once()
            assert mock_template.call_args[1]["chat_template"] == "chatml"

    # -- start_epoch=0 ignores resume_model --

    def test_start_epoch_zero_ignores_resume(self, tmp_path):
        """start_epoch=0 loads from model_name even if resume_model given."""
        from src.llm_finetune.unsloth_finetune import finetune

        with self._finetune_context(tmp_path) as m:
            finetune(
                train_json_paths=["t.json"],
                val_json_paths=["v.json"],
                model_name="test-model",
                perturb_type_names="test",
                max_seq_length=512,
                start_epoch=0,
                resume_model="/should/be/ignored",
            )
            load_call = m["flm"].from_pretrained.call_args
            # Should resolve "test-model" not the resume path
            assert load_call[1]["model_name"] == "test-model"

    # -- Training config export --

    def test_training_config_saved(self, tmp_path):
        """build_training_config and save_training_config are called."""
        from src.llm_finetune.unsloth_finetune import finetune

        with self._finetune_context(tmp_path) as m:
            finetune(
                train_json_paths=["t.json"],
                val_json_paths=["v.json"],
                model_name="test",
                perturb_type_names="test",
                max_seq_length=512,
            )
            m["refs"]["build_training_config"].assert_called_once()
            m["refs"]["save_training_config"].assert_called_once()

    # -- Non-default rank affects output path --

    def test_non_default_rank_in_output_path(self, tmp_path):
        """Rank != 16 adds _r{rank} suffix to output paths."""
        from src.llm_finetune.unsloth_finetune import finetune

        with self._finetune_context(tmp_path) as m:
            finetune(
                train_json_paths=["t.json"],
                val_json_paths=["v.json"],
                model_name="test",
                perturb_type_names="test",
                max_seq_length=512,
                rank=64,
            )
            # Check the save_pretrained call path contains _r64
            save_call = m["model"].save_pretrained.call_args
            assert "_r64" in save_call[0][0]

    def test_default_rank_no_suffix(self, tmp_path):
        """Rank == 16 uses no _r{rank} suffix in output paths."""
        from src.llm_finetune.unsloth_finetune import finetune

        with self._finetune_context(tmp_path) as m:
            finetune(
                train_json_paths=["t.json"],
                val_json_paths=["v.json"],
                model_name="test",
                perturb_type_names="test",
                max_seq_length=512,
                rank=16,
            )
            save_call = m["model"].save_pretrained.call_args
            # Should NOT have _r16 suffix
            assert "_r16" not in save_call[0][0]

    # -- debug=True does not crash --

    def test_debug_mode_runs_without_error(self, tmp_path):
        """Debug=True does not cause an error in the orchestration."""
        from src.llm_finetune.unsloth_finetune import finetune

        with self._finetune_context(tmp_path):
            finetune(
                train_json_paths=["t.json"],
                val_json_paths=["v.json"],
                model_name="test",
                perturb_type_names="test",
                max_seq_length=512,
                debug=True,
            )

    # -- wandb config includes expected keys --

    def test_wandb_config_keys(self, tmp_path):
        """wandb.init config dict contains key training parameters."""
        from src.llm_finetune.unsloth_finetune import finetune

        with self._finetune_context(tmp_path) as m:
            finetune(
                train_json_paths=["t.json"],
                val_json_paths=["v.json"],
                model_name="test-model",
                perturb_type_names="test",
                max_seq_length=512,
                batch_size=4,
                rank=32,
            )
            call_kwargs = m["refs"]["wandb"].init.call_args[1]
            config = call_kwargs["config"]
            assert config["model"] == "test-model"
            assert config["batch_size"] == 4
            assert config["rank"] == 32
            assert "max_seq_length" in config

    # -- wandb.finish called --

    def test_wandb_finish_called(self, tmp_path):
        """Wandb run.finish() is called after training."""
        from src.llm_finetune.unsloth_finetune import finetune

        with self._finetune_context(tmp_path) as m:
            finetune(
                train_json_paths=["t.json"],
                val_json_paths=["v.json"],
                model_name="test",
                perturb_type_names="test",
                max_seq_length=512,
            )
            # run.finish() is called on the run object returned by wandb.init
            run_mock = m["refs"]["wandb"].init.return_value
            run_mock.finish.assert_called_once()

    # -- Model save_pretrained_merged is called --

    def test_merged_model_saved(self, tmp_path):
        """Merged model is saved with merged_16bit method."""
        from src.llm_finetune.unsloth_finetune import finetune

        with self._finetune_context(tmp_path) as m:
            finetune(
                train_json_paths=["t.json"],
                val_json_paths=["v.json"],
                model_name="test",
                perturb_type_names="test",
                max_seq_length=512,
            )
            merge_call = m["model"].save_pretrained_merged.call_args
            assert merge_call[1]["save_method"] == "merged_16bit"

    # -- Batch sampler options forwarded --

    def test_batch_sampler_options_forwarded(self, tmp_path):
        """Batch sampler gets split_large_claims and ensure flags."""
        from src.llm_finetune.unsloth_finetune import finetune

        mock_sampler_cls = MagicMock()
        mock_trainer_cls = MagicMock(return_value=_make_mock_trainer())
        with self._finetune_context(
            tmp_path,
            EnhancedPerturbationBatchSampler=mock_sampler_cls,
            PerturbationAwareSFTTrainer=mock_trainer_cls,
        ):
            finetune(
                train_json_paths=["t.json"],
                val_json_paths=["v.json"],
                model_name="test",
                perturb_type_names="test",
                max_seq_length=512,
                batch_match_claim_id=True,
                split_large_claims=False,
                ensure_original_in_batch=False,
                ensure_perturbed_in_batch=True,
            )
            call_kwargs = mock_sampler_cls.call_args[1]
            assert call_kwargs["split_large_claims"] is False
            assert call_kwargs["ensure_original_in_batch"] is False
            assert call_kwargs["ensure_perturbed_in_batch"] is True

    # -- SFTConfig receives correct hyperparameters --

    def test_sft_config_hyperparameters(self, tmp_path):
        """SFTConfig receives batch_size and gradient_accumulation."""
        from src.llm_finetune.unsloth_finetune import finetune

        mock_sft_config = MagicMock(return_value=MagicMock())
        with self._finetune_context(tmp_path, SFTConfig=mock_sft_config):
            finetune(
                train_json_paths=["t.json"],
                val_json_paths=["v.json"],
                model_name="test",
                perturb_type_names="test",
                max_seq_length=512,
                batch_size=8,
                gradient_accumulation_steps=4,
                num_train_epochs=3,
            )
            call_kwargs = mock_sft_config.call_args[1]
            assert call_kwargs["per_device_train_batch_size"] == 8
            assert call_kwargs["gradient_accumulation_steps"] == 4
            assert call_kwargs["num_train_epochs"] == 3

    # -- Epoch calculation --

    def test_final_evaluation_called(self, tmp_path):
        """trainer.evaluate() is called after training."""
        from src.llm_finetune.unsloth_finetune import finetune

        with self._finetune_context(tmp_path) as m:
            finetune(
                train_json_paths=["t.json"],
                val_json_paths=["v.json"],
                model_name="test",
                perturb_type_names="test",
                max_seq_length=512,
            )
            m["trainer"].evaluate.assert_called_once()


# ---------------------------------------------------------------------------
# Additional edge-case tests for validation filtering
# ---------------------------------------------------------------------------


class TestValidationFilteringEdgeCases:
    """Edge cases for conversation filtering logic."""

    @staticmethod
    def _has_assistant_response(example):
        conv = example["conversations"]
        return len(conv) >= 2 and conv[-1]["role"] == "assistant"

    def test_single_message_conversation_filtered(self):
        """Conversation with only one message is filtered out."""
        ds = Dataset.from_list(
            [
                {
                    "claim_id": "c1",
                    "perturbation_type": "original_normalized",
                    "label": "True",
                    "conversations": [{"role": "user", "content": "hello"}],
                }
            ]
        )
        filtered = ds.filter(self._has_assistant_response)
        assert len(filtered) == 0

    def test_system_only_conversation_filtered(self):
        """Conversation ending in system role is filtered out."""
        ds = Dataset.from_list(
            [
                {
                    "claim_id": "c1",
                    "perturbation_type": "original_normalized",
                    "label": "True",
                    "conversations": [
                        {"role": "system", "content": "You are a checker."},
                        {"role": "user", "content": "check this"},
                        {"role": "system", "content": "Reconfigured."},
                    ],
                }
            ]
        )
        filtered = ds.filter(self._has_assistant_response)
        assert len(filtered) == 0

    def test_mixed_valid_and_invalid_keeps_valid(self):
        """Filter keeps valid rows and discards invalid ones."""
        rows = [
            {
                "claim_id": "good",
                "perturbation_type": "original_normalized",
                "label": "True",
                "conversations": [
                    {"role": "user", "content": "claim"},
                    {"role": "assistant", "content": "answer"},
                ],
            },
            {
                "claim_id": "bad",
                "perturbation_type": "original_normalized",
                "label": "True",
                "conversations": [
                    {"role": "user", "content": "claim"},
                ],
            },
        ]
        ds = Dataset.from_list(rows)
        filtered = ds.filter(self._has_assistant_response)
        assert len(filtered) == 1
        assert filtered[0]["claim_id"] == "good"
