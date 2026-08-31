import json

from src.data_preprocessing.raw_dataset_preprocessing.verdict_strip.hybrid_audit_report import (  # noqa: E501
    build_hybrid_audit_records,
)
from src.data_preprocessing.raw_dataset_preprocessing.verdict_strip.patch_hybrid_records import (  # noqa: E501
    extract_records,
    merge_subset,
    parse_keys,
)


def _write_split(dirpath, split, records):
    path = dirpath / f"{split}_claims_quantemp.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


def test_parse_and_extract(tmp_path):
    _write_split(tmp_path, "train", [
        {"datasplit": "train", "id": 1, "doc": "a"},
        {"datasplit": "train", "id": 2, "doc": "b"},
    ])
    _write_split(tmp_path, "test", [
        {"datasplit": "test", "id": 1, "doc": "c"},
    ])
    keys = parse_keys(["train:2", "test:1"])
    recs = extract_records(str(tmp_path), keys)
    assert {(r["datasplit"], r["id"]) for r in recs} == {
        ("train", 2), ("test", 1)
    }


def test_merge_patches_docs_and_difflog(tmp_path):
    hybrid = tmp_path / "hybrid"
    hybrid.mkdir()
    _write_split(hybrid, "train", [
        {"datasplit": "train", "id": 1, "doc": "old text with verdict"},
        {"datasplit": "train", "id": 2, "doc": "untouched"},
    ])
    stale = {"datasplit": "train", "id": 1, "kept": [],
             "rejected": [], "error": "timeout"}
    other = {"datasplit": "train", "id": 2, "kept": ["x"], "rejected": []}
    (hybrid / "llm_diff_log.jsonl").write_text(
        json.dumps(stale) + "\n" + json.dumps(other) + "\n",
        encoding="utf-8",
    )
    subset_records = [
        {"datasplit": "train", "id": 1, "doc": "clean text"}
    ]
    subset_difflog = [
        {"datasplit": "train", "id": 1, "kept": ["with verdict"],
         "rejected": [], "error": None}
    ]
    counts = merge_subset(subset_records, subset_difflog, str(hybrid))
    assert counts == {
        "docs_patched": 1, "difflog_dropped": 1, "difflog_appended": 1
    }
    recs = json.loads(
        (hybrid / "train_claims_quantemp.json").read_text()
    )
    assert recs[0]["doc"] == "clean text"
    assert recs[1]["doc"] == "untouched"
    entries = [
        json.loads(ln)
        for ln in (hybrid / "llm_diff_log.jsonl").read_text().splitlines()
    ]
    keys = [(e["datasplit"], e["id"]) for e in entries]
    assert keys.count(("train", 1)) == 1
    fresh = next(e for e in entries if e["id"] == 1)
    assert fresh["error"] is None and fresh["kept"] == ["with verdict"]


def test_build_audit_records_only_keys_includes_rejected_only():
    pristine = [{"datasplit": "t", "id": 1, "doc": "P", "claim": "c"}]
    hybrid = [{"datasplit": "t", "id": 1, "doc": "H"}]
    hy_log = [{"datasplit": "t", "id": 1, "kept": [],
               "rejected": ["a span"], "domain": "d"}]
    # default mode skips kept-empty entries
    assert build_hybrid_audit_records(pristine, hybrid, hy_log, []) == []
    # targeted mode includes them
    out = build_hybrid_audit_records(
        pristine, hybrid, hy_log, [], only_keys={("t", 1)}
    )
    assert len(out) == 1
    assert out[0]["rejected"] == ["a span"]
