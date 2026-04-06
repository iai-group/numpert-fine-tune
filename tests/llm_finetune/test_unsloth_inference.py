"""Tests for unsloth_inference module.

All GPU/model dependencies (unsloth, torch.cuda) are mocked so these tests run
on CPU-only machines.  We do NOT exercise actual model loading, generation, or
GPU code — only the orchestration logic, helper functions, and data-routing
decisions.
"""

import sys
import types
import json
import pytest
from unittest.mock import MagicMock, patch
from datasets import Dataset

# ---------------------------------------------------------------------------
# Stub heavy third-party packages BEFORE importing the module under test.
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
_unsloth = _ensure_module(
    "unsloth",
    {
        "FastLanguageModel": MagicMock(),
    },
)

# --- tqdm (prevent progress bars in test output) ---
_ensure_module("tqdm", {"tqdm": lambda x, **kw: x})

# Now safe to import the module under test
from src.llm_finetune.unsloth_inference import (  # noqa: E402
    get_model_name_arg,
    extract_label_from_response,
)


# ---------------------------------------------------------------------------
# Mock-data helpers
# ---------------------------------------------------------------------------


def _make_test_example(claim_id, perturb_type, label, claim_text="claim"):
    """Build a single example dict matching the inference pipeline."""
    return {
        "claim_id": claim_id,
        "perturbation_type": perturb_type,
        "label": label,
        "conversations": [
            {"role": "system", "content": "You are a fact-checker."},
            {
                "role": "user",
                "content": (f"Claim: {claim_text}\nEvidence: some doc"),
            },
        ],
    }


def create_mock_test_dataset(n_claims=5, perturbations_per_claim=2):
    """Create a mock HuggingFace Dataset for inference testing."""
    examples = []
    perturb_types = ["original_normalized", "perturb_type1", "perturb_type2"]
    for i in range(n_claims):
        cid = f"claim_{i:04d}"
        for j in range(perturbations_per_claim + 1):
            examples.append(
                _make_test_example(
                    cid,
                    perturb_types[j % len(perturb_types)],
                    "True" if j == 0 else "False",
                    claim_text=f"claim {i} variant {j}",
                )
            )
    return Dataset.from_list(examples)


@pytest.fixture
def small_test_dataset():
    """Small dataset fixture for inference tests."""
    return create_mock_test_dataset(n_claims=3, perturbations_per_claim=1)


# ---------------------------------------------------------------------------
# Tests for get_model_name_arg
# ---------------------------------------------------------------------------


class TestGetModelNameArg:
    """Test model name resolution (inference copy)."""

    def test_qwen3_8b(self):
        """Short Qwen3-8B resolves to unsloth path."""
        assert get_model_name_arg("Qwen3-8B") == "unsloth/Qwen3-8B"

    def test_qwen3_32b(self):
        """Short Qwen3-32B resolves to unsloth path."""
        assert get_model_name_arg("Qwen3-32B") == "unsloth/Qwen3-32B"

    def test_llama_3_2_3b(self):
        """Short llama-3.2-3b resolves to full Instruct path."""
        assert (
            get_model_name_arg("llama-3.2-3b")
            == "unsloth/Llama-3.2-3B-Instruct"
        )

    def test_llama_3_2_1b(self):
        """Short llama-3.2-1b resolves to full Instruct path."""
        assert (
            get_model_name_arg("llama-3.2-1b")
            == "unsloth/Llama-3.2-1B-Instruct"
        )

    def test_custom_model_passthrough(self):
        """Unknown model names pass through unchanged."""
        assert get_model_name_arg("my-org/my-model") == "my-org/my-model"

    def test_qwen3_8b_with_suffix(self):
        """Qwen3-8B with extra suffix still resolves to 8B."""
        assert get_model_name_arg("Qwen3-8B-bnb-4bit") == "unsloth/Qwen3-8B"

    def test_empty_string_passthrough(self):
        """Empty string passes through."""
        assert get_model_name_arg("") == ""

    def test_case_sensitive_no_match(self):
        """Model names are case-sensitive; lowercase passes through."""
        assert get_model_name_arg("qwen3-8b") == "qwen3-8b"

    def test_full_hf_path_qwen3_8b(self):
        """Full unsloth/Qwen3-8B path resolves to 8B, not 32B."""
        assert get_model_name_arg("unsloth/Qwen3-8B") == "unsloth/Qwen3-8B"

    def test_full_hf_path_qwen3_32b(self):
        """Full unsloth/Qwen3-32B path resolves to 32B."""
        assert get_model_name_arg("unsloth/Qwen3-32B") == "unsloth/Qwen3-32B"


# ---------------------------------------------------------------------------
# Tests for extract_label_from_response
# ---------------------------------------------------------------------------


class TestExtractLabelFromResponse:
    """Test label extraction from model responses."""

    def test_true_label(self):
        """Standard true label is parsed correctly."""
        result = extract_label_from_response('{"label": true}')
        assert result == '{"label": true}'

    def test_false_label(self):
        """Standard false label is parsed correctly."""
        result = extract_label_from_response('{"label": false}')
        assert result == '{"label": false}'

    def test_whitespace_stripped(self):
        """Leading/trailing whitespace is stripped before parsing."""
        result = extract_label_from_response('  {"label": true}  ')
        assert result == '{"label": true}'

    def test_invalid_json_returns_none(self):
        """Unparseable text returns None."""
        assert extract_label_from_response("not json at all") is None

    def test_empty_string_returns_none(self):
        """Empty string returns None."""
        assert extract_label_from_response("") is None

    def test_string_label_value_returns_none(self):
        """String 'true'/'false' instead of boolean returns None."""
        assert extract_label_from_response('{"label": "true"}') is None

    def test_missing_label_key_returns_none(self):
        """JSON without 'label' key returns None."""
        assert extract_label_from_response('{"answer": true}') is None

    def test_integer_label_returns_none(self):
        """Integer label value returns None (must be boolean)."""
        assert extract_label_from_response('{"label": 1}') is None

    def test_null_label_returns_none(self):
        """Null label value returns None."""
        assert extract_label_from_response('{"label": null}') is None

    def test_extra_keys_still_parsed(self):
        """Extra keys alongside 'label' are fine."""
        result = extract_label_from_response(
            '{"label": true, "confidence": 0.9}'
        )
        assert result == '{"label": true}'


class TestExtractLabelWithThinkingTokens:
    """Test label extraction with Qwen3-style thinking tokens."""

    def test_removes_thinking_block(self):
        """Content between <think> and </think> is removed."""
        text = "<think>Let me reason about this...</think>" '{"label": true}'
        result = extract_label_from_response(text)
        assert result == '{"label": true}'

    def test_multiline_thinking_block(self):
        """Multi-line thinking content is stripped."""
        text = (
            "<think>\nStep 1: Check numbers\n"
            "Step 2: Compare\n</think>\n"
            '{"label": false}'
        )
        result = extract_label_from_response(text)
        assert result == '{"label": false}'

    def test_unclosed_think_tag(self):
        """Unclosed <think> tag: takes content after </think>."""
        text = "<think>some reasoning"
        result = extract_label_from_response(text)
        # No </think> found, split produces original text, likely unparseable
        assert result is None

    def test_unclosed_think_with_answer_after(self):
        """Unclosed <think> with answer at end, no </think>."""
        text = '<think>reasoning{"label": true}'
        result = extract_label_from_response(text)
        # split("</think>")[-1] returns full text when no </think> exists
        # That full text starts with <think> which is not valid JSON
        assert result is None

    def test_empty_thinking_block(self):
        """Empty thinking block still works."""
        text = '<think></think>{"label": false}'
        result = extract_label_from_response(text)
        assert result == '{"label": false}'

    def test_only_thinking_no_answer(self):
        """Only thinking content, no answer, returns None."""
        text = "<think>I can't decide</think>"
        result = extract_label_from_response(text)
        assert result is None


# ---------------------------------------------------------------------------
# Tests for inference() orchestration
# ---------------------------------------------------------------------------


def _make_mock_inference_context(tmp_path, **extra_patches):
    """Return a context manager that patches all heavy deps for inference."""
    import contextlib

    mock_model = MagicMock()
    mock_tok = MagicMock()
    mock_tok.padding_side = "right"
    mock_tok.pad_token = None
    mock_tok.pad_token_id = 0
    mock_tok.eos_token = "<eos>"

    # Build a fake tensor that survives .cuda(), .shape, indexing, etc.
    mock_input_ids = MagicMock()
    mock_input_ids.cuda.return_value = mock_input_ids
    mock_input_ids.shape = (1, 5)
    mock_input_ids.__ne__ = MagicMock(return_value=mock_input_ids)
    mock_input_ids.long.return_value = mock_input_ids

    mock_attn = MagicMock()
    mock_attn.cuda.return_value = mock_attn

    def _apply_chat_template(conv, **kw):
        if kw.get("return_tensors") == "pt" or kw.get("tokenize", False):
            return {"input_ids": mock_input_ids}
        return " ".join(m["content"] for m in conv)

    mock_tok.apply_chat_template = MagicMock(side_effect=_apply_chat_template)
    mock_tok.return_value = {
        "input_ids": [1, 2, 3, 4, 5],
        "attention_mask": [1, 1, 1, 1, 1],
    }
    mock_tok.decode = MagicMock(return_value='{"label": true}')

    # model.generate returns a mock tensor: outputs[0][input_length:]
    mock_output_slice = MagicMock()
    mock_output_row = MagicMock()
    mock_output_row.__getitem__ = MagicMock(return_value=mock_output_slice)
    mock_outputs = MagicMock()
    mock_outputs.__getitem__ = MagicMock(return_value=mock_output_row)
    mock_model.generate = MagicMock(return_value=mock_outputs)

    mock_flm = MagicMock()
    mock_flm.from_pretrained.return_value = (mock_model, mock_tok)
    mock_flm.for_inference = MagicMock()

    test_ds = create_mock_test_dataset(n_claims=3, perturbations_per_claim=1)

    mock_report = MagicMock()
    mock_report.to_dict.return_value = {"overall": {"accuracy": 0.9}}

    patches = {
        "FastLanguageModel": mock_flm,
        "load_split": MagicMock(return_value=test_ds),
        "build_inference_config": MagicMock(return_value={}),
        "save_inference_config": MagicMock(
            return_value=str(tmp_path / "config.json")
        ),
        "build_result_records": MagicMock(return_value=[]),
        "compute_metrics": MagicMock(return_value=mock_report),
        "print_metrics_report": MagicMock(),
        "save_metrics_report": MagicMock(
            return_value=str(tmp_path / "metrics.json")
        ),
        "save_parse_failures": MagicMock(return_value=None),
        "save_predictions_csv": MagicMock(
            return_value=str(tmp_path / "preds.csv")
        ),
        "prefix": str(tmp_path) + "/",
    }
    patches.update(extra_patches)

    prefix_str = "src.llm_finetune.unsloth_inference."

    @contextlib.contextmanager
    def ctx():
        stack = contextlib.ExitStack()
        refs = {}
        with stack:
            for name, value in patches.items():
                p = stack.enter_context(patch(prefix_str + name, value))
                refs[name] = p
            yield {
                "model": mock_model,
                "tokenizer": mock_tok,
                "flm": mock_flm,
                "refs": refs,
                "test_ds": test_ds,
            }

    return ctx()


class TestInference:
    """Test inference() orchestration with all heavy deps mocked."""

    def test_basic_inference_runs(self, tmp_path):
        """Inference() completes without error."""
        from src.llm_finetune.unsloth_inference import inference

        with _make_mock_inference_context(tmp_path) as m:
            inference(
                test_json_paths=["test.json"],
                model_name="test-model",
                max_seq_length=512,
                output_dir=str(tmp_path / "output"),
            )
            m["flm"].from_pretrained.assert_called_once()
            m["flm"].for_inference.assert_called_once()

    def test_model_loaded_from_hf(self, tmp_path):
        """Default start_epoch=0 loads from HF hub."""
        from src.llm_finetune.unsloth_inference import inference

        with _make_mock_inference_context(tmp_path) as m:
            inference(
                test_json_paths=["test.json"],
                model_name="test-model",
                max_seq_length=512,
                output_dir=str(tmp_path / "output"),
            )
            load_call = m["flm"].from_pretrained.call_args
            assert load_call[1]["model_name"] == "test-model"

    def test_local_dir_mode(self, tmp_path):
        """If model_name is an existing directory, use it directly."""
        from src.llm_finetune.unsloth_inference import inference

        # Create a directory so os.path.isdir returns True
        model_dir = tmp_path / "local_model"
        model_dir.mkdir()

        with _make_mock_inference_context(tmp_path) as m:
            inference(
                test_json_paths=["test.json"],
                model_name=str(model_dir),
                max_seq_length=512,
                output_dir=str(tmp_path / "output"),
            )
            load_call = m["flm"].from_pretrained.call_args
            assert load_call[1]["model_name"] == str(model_dir)

    def test_inference_config_saved(self, tmp_path):
        """build_inference_config and save_inference_config are called."""
        from src.llm_finetune.unsloth_inference import inference

        with _make_mock_inference_context(tmp_path) as m:
            inference(
                test_json_paths=["test.json"],
                model_name="test",
                max_seq_length=512,
                output_dir=str(tmp_path / "output"),
            )
            m["refs"]["build_inference_config"].assert_called_once()
            m["refs"]["save_inference_config"].assert_called_once()

    def test_metrics_computed_and_saved(self, tmp_path):
        """Metrics pipeline is fully invoked."""
        from src.llm_finetune.unsloth_inference import inference

        with _make_mock_inference_context(tmp_path) as m:
            inference(
                test_json_paths=["test.json"],
                model_name="test",
                max_seq_length=512,
                output_dir=str(tmp_path / "output"),
            )
            m["refs"]["build_result_records"].assert_called_once()
            m["refs"]["compute_metrics"].assert_called_once()
            m["refs"]["print_metrics_report"].assert_called_once()
            m["refs"]["save_metrics_report"].assert_called_once()
            m["refs"]["save_predictions_csv"].assert_called_once()

    def test_output_dir_created(self, tmp_path):
        """Inference creates output directory for predictions."""
        from src.llm_finetune.unsloth_inference import inference

        out = tmp_path / "output"
        with _make_mock_inference_context(tmp_path):
            inference(
                test_json_paths=["test.json"],
                model_name="test-model",
                max_seq_length=512,
                output_dir=str(out),
            )
        # inference creates pred_dir under output_dir
        assert out.exists()

    def test_rank_affects_output_path(self, tmp_path):
        """Non-default rank includes _r{rank} in output path."""
        from src.llm_finetune.unsloth_inference import inference

        out = tmp_path / "output"
        with _make_mock_inference_context(tmp_path):
            inference(
                test_json_paths=["test.json"],
                model_name="test-model",
                rank=64,
                max_seq_length=512,
                output_dir=str(out),
            )
        # Check that a directory with _r64 was created
        subdirs = [d.name for d in out.iterdir() if d.is_dir()]
        assert any("_r64" in d for d in subdirs)

    def test_default_rank_no_suffix(self, tmp_path):
        """Rank == 16 produces no _r suffix in output path."""
        from src.llm_finetune.unsloth_inference import inference

        out = tmp_path / "output"
        with _make_mock_inference_context(tmp_path):
            inference(
                test_json_paths=["test.json"],
                model_name="test-model",
                rank=16,
                max_seq_length=512,
                output_dir=str(out),
            )
        subdirs = [d.name for d in out.iterdir() if d.is_dir()]
        assert all("_r16" not in d for d in subdirs)

    def test_inference_details_json_written(self, tmp_path):
        """Raw inference details are written to a JSON file."""
        from src.llm_finetune.unsloth_inference import inference

        out = tmp_path / "output"
        with _make_mock_inference_context(tmp_path):
            inference(
                test_json_paths=["test.json"],
                model_name="test-model",
                max_seq_length=512,
                output_dir=str(out),
            )
        # Find inference_details.json somewhere under output
        json_files = list(out.rglob("inference_details.json"))
        assert len(json_files) == 1

        with open(json_files[0]) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) > 0

    def test_load_split_called(self, tmp_path):
        """Test data loading function is called with correct args."""
        from src.llm_finetune.unsloth_inference import inference

        with _make_mock_inference_context(tmp_path) as m:
            inference(
                test_json_paths=["test.json"],
                model_name="test",
                max_seq_length=512,
                expand_perturbations=False,
                output_dir=str(tmp_path / "output"),
            )
            load_call = m["refs"]["load_split"].call_args
            assert load_call[1]["expand_perturbations"] is False


class TestInferenceModelResolution:
    """Test model path resolution logic in inference()."""

    def test_epoch_checkpoint_not_found_falls_back(self, tmp_path):
        """When checkpoint dir doesn't exist, falls back to HF name."""
        from src.llm_finetune.unsloth_inference import inference

        with _make_mock_inference_context(tmp_path) as m:
            inference(
                test_json_paths=["test.json"],
                model_name="Qwen3-8B",
                start_epoch=3,
                max_seq_length=512,
                output_dir=str(tmp_path / "output"),
            )
            load_call = m["flm"].from_pretrained.call_args
            # Falls back to get_model_name_arg("Qwen3-8B")
            assert load_call[1]["model_name"] == "unsloth/Qwen3-8B"

    def test_epoch_checkpoint_found_uses_local(self, tmp_path):
        """When checkpoint dir exists, uses it."""
        from src.llm_finetune.unsloth_inference import inference

        # Create the expected checkpoint directory
        checkpoint = (
            tmp_path
            / "tuned_models"
            / "epoch-2"
            / "test-model"
            / "lora_numpert_r32"
        )
        checkpoint.mkdir(parents=True)

        with _make_mock_inference_context(tmp_path) as m:
            inference(
                test_json_paths=["test.json"],
                model_name="test-model",
                rank=32,
                start_epoch=2,
                max_seq_length=512,
                output_dir=str(tmp_path / "output"),
            )
            load_call = m["flm"].from_pretrained.call_args
            assert load_call[1]["model_name"] == str(checkpoint)

    def test_qwen_name_resolved_via_alias(self, tmp_path):
        """Qwen3-8B alias is resolved when epoch is 0."""
        from src.llm_finetune.unsloth_inference import inference

        with _make_mock_inference_context(tmp_path) as m:
            inference(
                test_json_paths=["test.json"],
                model_name="Qwen3-8B",
                start_epoch=0,
                max_seq_length=512,
                output_dir=str(tmp_path / "output"),
            )
            load_call = m["flm"].from_pretrained.call_args
            assert load_call[1]["model_name"] == "unsloth/Qwen3-8B"


# ---------------------------------------------------------------------------
# Tests for mock test dataset integrity
# ---------------------------------------------------------------------------


class TestMockTestDatasetIntegrity:
    """Verify mock dataset helpers produce valid structures."""

    def test_columns_present(self, small_test_dataset):
        """Dataset has the required columns."""
        expected = {
            "claim_id",
            "perturbation_type",
            "label",
            "conversations",
        }
        assert expected.issubset(set(small_test_dataset.column_names))

    def test_dataset_size(self, small_test_dataset):
        """3 claims * (1 original + 1 perturbation) = 6 rows."""
        assert len(small_test_dataset) == 6

    def test_conversations_have_no_assistant(self, small_test_dataset):
        """Inference data has system + user but no assistant."""
        for i in range(len(small_test_dataset)):
            conv = small_test_dataset[i]["conversations"]
            assert conv[-1]["role"] == "user"

    def test_label_values(self, small_test_dataset):
        """All labels are valid True/False strings."""
        labels = set(small_test_dataset["label"])
        assert labels.issubset({"True", "False"})

    def test_claim_ids_grouped(self, small_test_dataset):
        """Each claim_id appears the expected number of times."""
        from collections import Counter

        counter = Counter(small_test_dataset["claim_id"])
        for count in counter.values():
            assert count == 2  # 1 original + 1 perturbation

    def test_perturbation_types_present(self, small_test_dataset):
        """Multiple perturbation types are present."""
        types = set(small_test_dataset["perturbation_type"])
        assert "original_normalized" in types
        assert len(types) > 1


# ---------------------------------------------------------------------------
# Tests for unperturbed-claims inference mode
# ---------------------------------------------------------------------------


class TestUnperturbedClaimsInference:
    """Test --inference_unperturbed_claims flag behaviour in inference()."""

    def test_unperturbed_forces_expand_false(self, tmp_path):
        """inference_unperturbed_claims=True forces
        expand_perturbations=False."""
        from src.llm_finetune.unsloth_inference import inference

        with _make_mock_inference_context(tmp_path) as m:
            inference(
                test_json_paths=["test.json"],
                model_name="test",
                max_seq_length=512,
                output_dir=str(tmp_path / "output"),
                inference_unperturbed_claims=True,
            )
            load_call = m["refs"]["load_split"].call_args
            assert load_call[1]["expand_perturbations"] is False

    def test_unperturbed_output_files_prefixed(self, tmp_path):
        """inference_unperturbed_claims=True produces unperturbed_ prefixed
        output files."""
        from src.llm_finetune.unsloth_inference import inference

        out = tmp_path / "output"
        with _make_mock_inference_context(tmp_path) as m:
            inference(
                test_json_paths=["test.json"],
                model_name="test-model",
                max_seq_length=512,
                output_dir=str(out),
                inference_unperturbed_claims=True,
            )
            # metrics saved with unperturbed_ prefix
            save_call = m["refs"]["save_metrics_report"].call_args
            assert save_call[1]["filename"] == "unperturbed_metrics.json"

            # predictions csv saved with unperturbed_ prefix
            csv_call = m["refs"]["save_predictions_csv"].call_args
            assert csv_call[1]["filename"] == "unperturbed_predictions.csv"

            # parse failures saved with unperturbed_ prefix
            fail_call = m["refs"]["save_parse_failures"].call_args
            assert fail_call[1]["filename"] == "unperturbed_parse_failures.json"

            # inference config saved with unperturbed_ prefix
            cfg_call = m["refs"]["save_inference_config"].call_args
            assert (
                cfg_call[1]["filename"] == "unperturbed_inference_config.json"
            )

    def test_unperturbed_inference_details_prefixed(self, tmp_path):
        """inference_unperturbed_claims=True writes
        unperturbed_inference_details.json."""
        from src.llm_finetune.unsloth_inference import inference

        out = tmp_path / "output"
        with _make_mock_inference_context(tmp_path):
            inference(
                test_json_paths=["test.json"],
                model_name="test-model",
                max_seq_length=512,
                output_dir=str(out),
                inference_unperturbed_claims=True,
            )
        json_files = list(out.rglob("unperturbed_inference_details.json"))
        assert len(json_files) == 1

    def test_non_unperturbed_no_prefix(self, tmp_path):
        """Without the flag, output files have no prefix."""
        from src.llm_finetune.unsloth_inference import inference

        with _make_mock_inference_context(tmp_path) as m:
            inference(
                test_json_paths=["test.json"],
                model_name="test",
                max_seq_length=512,
                output_dir=str(tmp_path / "output"),
                inference_unperturbed_claims=False,
            )
            save_call = m["refs"]["save_metrics_report"].call_args
            assert save_call[1]["filename"] == "metrics.json"

            cfg_call = m["refs"]["save_inference_config"].call_args
            assert cfg_call[1]["filename"] == "inference_config.json"


# ---------------------------------------------------------------------------
# Tests for CLI argument parsing (parse_file_config)
# ---------------------------------------------------------------------------


class TestCLIArgParsing:
    """Test argparse / FileLoadConfig parsing for inference."""

    def test_parse_file_config_simple(self):
        """Simple path without types."""
        from src.llm_finetune.utils.data_loading.perturbation_types import (
            parse_file_config,
        )

        cfg = parse_file_config("data/test.json")
        assert cfg.path == "data/test.json"
        assert cfg.perturb_types is None

    def test_parse_file_config_with_types(self):
        """Path with colon-separated types."""
        from src.llm_finetune.utils.data_loading.perturbation_types import (
            parse_file_config,
            PerturbationType,
        )

        cfg = parse_file_config("data/test.json:type1,type2")
        assert cfg.path == "data/test.json"
        assert cfg.perturb_types == [
            PerturbationType.PERTURB_TYPE1,
            PerturbationType.PERTURB_TYPE2,
        ]

    def test_parse_file_config_invalid_type_raises(self):
        """Invalid perturbation type raises ValueError."""
        from src.llm_finetune.utils.data_loading.perturbation_types import (
            parse_file_config,
        )

        with pytest.raises(ValueError, match="Unknown perturbation type"):
            parse_file_config("data/test.json:nonexistent_type")


# ---------------------------------------------------------------------------
# Helpers for deduplication tests
# ---------------------------------------------------------------------------


def _make_test_example_with_source(
    claim_id, perturb_type, label, source_file, claim_text="claim"
):
    """Build a single example dict with ``source_file``."""
    return {
        "claim_id": claim_id,
        "perturbation_type": perturb_type,
        "label": label,
        "source_file": source_file,
        "conversations": [
            {"role": "system", "content": "You are a fact-checker."},
            {
                "role": "user",
                "content": f"Claim: {claim_text}\nEvidence: some doc",
            },
        ],
    }


def _create_dataset_with_duplicates():
    """Create a dataset simulating multiple source files sharing claims.

    Scenario: two source files (``range_True.json`` and
    ``approximation_True.json``) each contribute an
    ``original_normalized`` entry for the same claim, plus their own
    perturbation entries (which may share the same type but have
    different claim text).  Result: 8 rows, 6 unique evaluations
    (only original_normalized duplicates are skipped).
    """
    examples = [
        # range_True.json entries
        _make_test_example_with_source(
            "c1",
            "original_normalized",
            "True",
            "range_True.json",
            "claim one",
        ),
        _make_test_example_with_source(
            "c1",
            "perturb_type1",
            "True",
            "range_True.json",
            "claim one range perturb",
        ),
        # approximation_True.json entries
        _make_test_example_with_source(
            "c1",
            "original_normalized",
            "True",
            "approximation_True.json",
            "claim one",
        ),
        _make_test_example_with_source(
            "c1",
            "perturb_type1",
            "True",
            "approximation_True.json",
            "claim one approx perturb",
        ),
        _make_test_example_with_source(
            "c1",
            "perturb_type2",
            "False",
            "approximation_True.json",
            "claim one approx type2",
        ),
        # A second claim with the same pattern
        _make_test_example_with_source(
            "c2",
            "original_normalized",
            "False",
            "range_True.json",
            "claim two",
        ),
        _make_test_example_with_source(
            "c2",
            "original_normalized",
            "False",
            "approximation_True.json",
            "claim two",
        ),
        _make_test_example_with_source(
            "c2",
            "perturb_type1",
            "False",
            "range_True.json",
            "claim two range perturb",
        ),
    ]
    return Dataset.from_list(examples)


# ---------------------------------------------------------------------------
# Tests for inference deduplication
# ---------------------------------------------------------------------------


class TestInferenceDeduplication:
    """Verify that duplicate original_normalized entries are evaluated only once
    and the result is reused.

    Perturbed entries with the same type from different files must always be
    evaluated.
    """

    def test_model_generate_called_only_for_unique(self, tmp_path):
        """model.generate is called for every perturbed entry plus one per
        unique original_normalized claim_id."""
        from src.llm_finetune.unsloth_inference import inference

        ds = _create_dataset_with_duplicates()
        # 8 rows total, 2 duplicate original_normalized → 6 unique evals
        with _make_mock_inference_context(
            tmp_path,
            load_split=MagicMock(return_value=ds),
        ) as m:
            inference(
                test_json_paths=["test.json"],
                model_name="test",
                max_seq_length=512,
                output_dir=str(tmp_path / "output"),
            )
            assert m["model"].generate.call_count == 6

    def test_all_entries_in_output(self, tmp_path):
        """inference_details.json contains an entry for every row, including
        duplicates."""
        from src.llm_finetune.unsloth_inference import inference

        ds = _create_dataset_with_duplicates()
        out = tmp_path / "output"
        with _make_mock_inference_context(
            tmp_path,
            load_split=MagicMock(return_value=ds),
        ):
            inference(
                test_json_paths=["test.json"],
                model_name="test",
                max_seq_length=512,
                output_dir=str(out),
            )
        json_files = list(out.rglob("inference_details.json"))
        assert len(json_files) == 1
        with open(json_files[0]) as f:
            details = json.load(f)
        assert len(details) == 8

    def test_duplicates_have_reused_marker(self, tmp_path):
        """Only duplicate original_normalized entries are tagged with
        ``reused_from_index``."""
        from src.llm_finetune.unsloth_inference import inference

        ds = _create_dataset_with_duplicates()
        out = tmp_path / "output"
        with _make_mock_inference_context(
            tmp_path,
            load_split=MagicMock(return_value=ds),
        ):
            inference(
                test_json_paths=["test.json"],
                model_name="test",
                max_seq_length=512,
                output_dir=str(out),
            )
        json_files = list(out.rglob("inference_details.json"))
        with open(json_files[0]) as f:
            details = json.load(f)

        reused = [d for d in details if "reused_from_index" in d]
        not_reused = [d for d in details if "reused_from_index" not in d]
        assert len(reused) == 2  # two duplicate original_normalized
        assert len(not_reused) == 6
        # All reused entries must be original_normalized
        for d in reused:
            assert d["perturbation_type"] == "original_normalized"

    def test_duplicate_preserves_source_file(self, tmp_path):
        """Reused entries retain their own ``source_file``, not the
        original's."""
        from src.llm_finetune.unsloth_inference import inference

        ds = _create_dataset_with_duplicates()
        out = tmp_path / "output"
        with _make_mock_inference_context(
            tmp_path,
            load_split=MagicMock(return_value=ds),
        ):
            inference(
                test_json_paths=["test.json"],
                model_name="test",
                max_seq_length=512,
                output_dir=str(out),
            )
        json_files = list(out.rglob("inference_details.json"))
        with open(json_files[0]) as f:
            details = json.load(f)

        reused = [d for d in details if "reused_from_index" in d]
        # The duplicates come from approximation_True.json
        source_files = {d["source_file"] for d in reused}
        assert "approximation_True.json" in source_files

    def test_no_duplicates_means_no_reuse(self, tmp_path):
        """When all entries are unique, no reuse markers appear and
        model.generate is called for every entry."""
        from src.llm_finetune.unsloth_inference import inference

        # Default dataset has no duplicates
        ds = create_mock_test_dataset(n_claims=3, perturbations_per_claim=1)
        with _make_mock_inference_context(
            tmp_path,
            load_split=MagicMock(return_value=ds),
        ) as m:
            inference(
                test_json_paths=["test.json"],
                model_name="test",
                max_seq_length=512,
                output_dir=str(tmp_path / "output"),
            )
            assert m["model"].generate.call_count == len(ds)

    def test_same_perturb_type_different_files_both_evaluated(self, tmp_path):
        """perturb_type1 from two files for the same claim_id must both go
        through model.generate (different claim text)."""
        from src.llm_finetune.unsloth_inference import inference

        examples = [
            _make_test_example_with_source(
                "c1",
                "perturb_type1",
                "True",
                "range_True.json",
                "claim one range perturb",
            ),
            _make_test_example_with_source(
                "c1",
                "perturb_type1",
                "True",
                "approximation_True.json",
                "claim one approx perturb",
            ),
        ]
        ds = Dataset.from_list(examples)
        out = tmp_path / "output"

        with _make_mock_inference_context(
            tmp_path,
            load_split=MagicMock(return_value=ds),
        ) as m:
            inference(
                test_json_paths=["test.json"],
                model_name="test",
                max_seq_length=512,
                output_dir=str(out),
            )
            # Both must be evaluated — no deduplication
            assert m["model"].generate.call_count == 2

        json_files = list(out.rglob("inference_details.json"))
        with open(json_files[0]) as f:
            details = json.load(f)
        # No reused entries
        assert all("reused_from_index" not in d for d in details)

    def test_duplicate_shares_predicted_label(self, tmp_path):
        """Reused entry has the same ``predicted_label`` as original."""
        from src.llm_finetune.unsloth_inference import inference

        ds = _create_dataset_with_duplicates()
        out = tmp_path / "output"
        with _make_mock_inference_context(
            tmp_path,
            load_split=MagicMock(return_value=ds),
        ):
            inference(
                test_json_paths=["test.json"],
                model_name="test",
                max_seq_length=512,
                output_dir=str(out),
            )
        json_files = list(out.rglob("inference_details.json"))
        with open(json_files[0]) as f:
            details = json.load(f)

        by_index = {d["index"]: d for d in details}
        for d in details:
            if "reused_from_index" in d:
                orig = by_index[d["reused_from_index"]]
                assert d["predicted_label"] == orig["predicted_label"]
                assert d["raw_response"] == orig["raw_response"]


# ---------------------------------------------------------------------------
# Tests for deduplication key logic (unit-level, no model needed)
# ---------------------------------------------------------------------------


class TestDeduplicationKeyLogic:
    """Verify that only ``original_normalized`` entries are deduplicated by
    ``claim_id``.

    Perturbed entries are never deduplicated, even if
    they share the same ``(claim_id, perturbation_type)``.
    """

    def test_original_normalized_duplicates_detected(self):
        """Two original_normalized rows for the same claim_id are duplicates."""
        ds = _create_dataset_with_duplicates()
        # c1/original_normalized at index 0, c1/original_normalized at index 2
        assert ds[0]["perturbation_type"] == "original_normalized"
        assert ds[2]["perturbation_type"] == "original_normalized"
        assert ds[0]["claim_id"] == ds[2]["claim_id"]

    def test_same_perturb_type_different_files_not_duplicated(self):
        """perturb_type1 for c1 from two files must both be kept."""
        ds = _create_dataset_with_duplicates()
        # Index 1: c1/perturb_type1/range_True.json
        # Index 3: c1/perturb_type1/approximation_True.json
        assert ds[1]["perturbation_type"] == "perturb_type1"
        assert ds[3]["perturbation_type"] == "perturb_type1"
        assert ds[1]["claim_id"] == ds[3]["claim_id"]
        # But they have different claim text, so must NOT be deduped
        assert ds[1]["source_file"] != ds[3]["source_file"]

    def test_first_occurrence_wins(self):
        """The dedup map records the first index for each original_normalized
        claim_id."""
        ds = _create_dataset_with_duplicates()
        seen: dict = {}
        unique_indices: list = []
        duplicate_map: dict = {}

        for i in range(len(ds)):
            if ds[i]["perturbation_type"] == "original_normalized":
                cid = ds[i]["claim_id"]
                if cid not in seen:
                    seen[cid] = i
                    unique_indices.append(i)
                else:
                    duplicate_map[i] = seen[cid]
            else:
                unique_indices.append(i)

        # 8 rows, 2 duplicate original_normalized → 6 unique
        assert unique_indices == [0, 1, 3, 4, 5, 7]
        assert duplicate_map == {2: 0, 6: 5}

    def test_empty_dataset_produces_no_duplicates(self):
        """An empty dataset has no unique indices or duplicates."""
        ds = Dataset.from_list([])
        seen: dict = {}
        unique_indices: list = []
        duplicate_map: dict = {}

        for i in range(len(ds)):
            if ds[i]["perturbation_type"] == "original_normalized":
                cid = ds[i]["claim_id"]
                if cid not in seen:
                    seen[cid] = i
                    unique_indices.append(i)
                else:
                    duplicate_map[i] = seen[cid]
            else:
                unique_indices.append(i)

        assert unique_indices == []
        assert duplicate_map == {}

    def test_all_unique_no_duplicates(self):
        """When every row has a distinct key, duplicate_map is empty."""
        ds = create_mock_test_dataset(n_claims=3, perturbations_per_claim=1)
        seen: dict = {}
        duplicate_map: dict = {}
        for i in range(len(ds)):
            if ds[i]["perturbation_type"] == "original_normalized":
                cid = ds[i]["claim_id"]
                if cid in seen:
                    duplicate_map[i] = seen[cid]
                else:
                    seen[cid] = i
        assert duplicate_map == {}

    def test_perturbed_entries_never_deduped_even_with_same_key(self):
        """Two perturb_type1 entries for the same claim_id from different source
        files must both be evaluated."""
        examples = [
            _make_test_example_with_source(
                "c1",
                "perturb_type1",
                "True",
                "file_a.json",
                "claim one file-a",
            ),
            _make_test_example_with_source(
                "c1",
                "perturb_type1",
                "True",
                "file_b.json",
                "claim one file-b",
            ),
        ]
        ds = Dataset.from_list(examples)

        seen: dict = {}
        unique_indices: list = []
        duplicate_map: dict = {}
        for i in range(len(ds)):
            if ds[i]["perturbation_type"] == "original_normalized":
                cid = ds[i]["claim_id"]
                if cid not in seen:
                    seen[cid] = i
                    unique_indices.append(i)
                else:
                    duplicate_map[i] = seen[cid]
            else:
                unique_indices.append(i)

        # Both must be kept
        assert unique_indices == [0, 1]
        assert duplicate_map == {}


# ---------------------------------------------------------------------------
# End-to-end deduplication → metrics tests
# ---------------------------------------------------------------------------


class TestDeduplicationMetricsIntegration:
    """Verify that reused inference results flow correctly through
    ``build_result_records`` → ``compute_metrics`` so that per-source-file
    breakdowns report the right counts."""

    def _run_dedup_pipeline(self):
        """Simulate the dedup + result-building pipeline and return the metrics
        report.

        Uses ``_create_dataset_with_duplicates()`` (8 rows, 6 unique).
        """
        from src.llm_finetune.utils.inference.metrics import (
            build_result_records,
            compute_metrics,
        )

        ds = _create_dataset_with_duplicates()

        # Simulate inference results: all predictions are correct
        raw_outputs = []
        for i in range(len(ds)):
            label = ds[i]["label"]
            pred = '{"label": true}' if label == "True" else '{"label": false}'
            entry = {
                "index": i,
                "claim_id": ds[i]["claim_id"],
                "perturbation_type": ds[i]["perturbation_type"],
                "label": label,
                "predicted_label": pred,
                "parsed_successfully": True,
                "raw_response": pred,
                "full_prompt": "prompt",
            }
            if "source_file" in ds.column_names:
                entry["source_file"] = ds[i]["source_file"]
            raw_outputs.append(entry)

        records = build_result_records(ds, raw_outputs)
        report = compute_metrics(records)
        return report, ds

    def test_overall_count_includes_all_rows(self):
        """Overall metrics count all 8 rows, not just the 6 unique."""
        report, ds = self._run_dedup_pipeline()
        assert report.overall.total == len(ds)  # 8

    def test_per_source_file_counts(self):
        """Each source file reports its own row count."""
        report, _ = self._run_dedup_pipeline()
        assert "range_True.json" in report.per_source_file
        assert "approximation_True.json" in report.per_source_file
        assert report.per_source_file["range_True.json"].total == 4
        assert report.per_source_file["approximation_True.json"].total == 4

    def test_per_file_and_type_original_normalized(self):
        """Both source files report their own original_normalized count."""
        report, _ = self._run_dedup_pipeline()
        range_orig = report.per_file_and_type.get(
            "range_True.json | original_normalized"
        )
        approx_orig = report.per_file_and_type.get(
            "approximation_True.json | original_normalized"
        )
        assert range_orig is not None
        assert approx_orig is not None
        assert range_orig.total == 2  # c1 + c2
        assert approx_orig.total == 2  # c1 + c2

    def test_per_perturbation_type_overall(self):
        """Perturbation-type grouping sums original_normalized duplicates
        together."""
        report, _ = self._run_dedup_pipeline()
        orig = report.per_perturbation_type.get("original_normalized")
        assert orig is not None
        # 4 original_normalized rows (c1 x2, c2 x2)
        assert orig.total == 4

    def test_perturbed_entries_all_counted_per_file(self):
        """Same perturbation type from different files counts separately per
        source file."""
        report, _ = self._run_dedup_pipeline()
        range_t1 = report.per_file_and_type.get(
            "range_True.json | perturb_type1"
        )
        approx_t1 = report.per_file_and_type.get(
            "approximation_True.json | perturb_type1"
        )
        assert range_t1 is not None
        assert approx_t1 is not None
        # c1/perturb_type1 from range, c2/perturb_type1 from range
        assert range_t1.total == 2
        # c1/perturb_type1 from approximation
        assert approx_t1.total == 1

    def test_three_source_files_same_claim(self):
        """Three source files sharing one claim: metrics count all 3."""
        from src.llm_finetune.utils.inference.metrics import (
            build_result_records,
            compute_metrics,
        )

        examples = [
            _make_test_example_with_source(
                "c1",
                "original_normalized",
                "True",
                "file_a.json",
                "claim one",
            ),
            _make_test_example_with_source(
                "c1",
                "original_normalized",
                "True",
                "file_b.json",
                "claim one",
            ),
            _make_test_example_with_source(
                "c1",
                "original_normalized",
                "True",
                "file_c.json",
                "claim one",
            ),
        ]
        ds = Dataset.from_list(examples)

        raw_outputs = []
        for i in range(len(ds)):
            raw_outputs.append(
                {
                    "index": i,
                    "claim_id": "c1",
                    "perturbation_type": "original_normalized",
                    "label": "True",
                    "predicted_label": '{"label": true}',
                    "parsed_successfully": True,
                    "raw_response": '{"label": true}',
                    "full_prompt": "prompt",
                    "source_file": ds[i]["source_file"],
                }
            )

        records = build_result_records(ds, raw_outputs)
        report = compute_metrics(records)

        assert report.overall.total == 3
        assert len(report.per_source_file) == 3
        for src in ["file_a.json", "file_b.json", "file_c.json"]:
            assert report.per_source_file[src].total == 1
