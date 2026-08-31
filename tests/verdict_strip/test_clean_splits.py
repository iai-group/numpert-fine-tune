import json
import sys

from src.data_preprocessing.raw_dataset_preprocessing.verdict_strip.clean_splits import (  # noqa: E501
    clean_records,
    main,
)
from src.data_preprocessing.raw_dataset_preprocessing.verdict_strip.rules import (  # noqa: E501
    DROP_KEYS,
    DROP_REASONS,
)


def test_clean_records_produces_cleaned_docs_and_difflog():
    records = [
        {
            "id": 0, "datasplit": "test", "url": "https://www.indiatoday.in/x",
            "doc": "Evidence here. JHOOTH BOLE KAUVA KAATE The number of crows "
                   "determines the intensity of the lie.",
        },
        {
            "id": 1, "datasplit": "test", "url": "https://www.reuters.com/y",
            "doc": "Clean evidence, unchanged.",
        },
    ]
    cleaned, difflog = clean_records(records)
    assert cleaned[0]["doc"] == "Evidence here."
    assert cleaned[1]["doc"] == "Clean evidence, unchanged."  # untouched
    assert len(difflog) == 1
    assert difflog[0]["id"] == 0
    assert difflog[0]["datasplit"] == "test"
    assert difflog[0]["domain"] == "indiatoday.in"
    assert difflog[0]["rules"] == ["jhooth_tagline"]


def test_missing_url_is_untouched():
    records = [{"id": 0, "datasplit": "test", "doc": "No url, leave alone."}]
    cleaned, difflog = clean_records(records)
    assert cleaned[0]["doc"] == "No url, leave alone."
    assert difflog == []


# =====================================================================
# Record drops (DROP_KEYS): badly-scraped / compilation records are
# removed wholesale from the cleaned splits, before any stripping.
# =====================================================================

# Real corpus doc of train:6419 (thequint, 167 chars): archive-links
# parenthetical + verdict sentence, no evidence body.
_QUINT_6419_DOC = (
    "(Archives of more claims can be seen here, here and here.)But...?: "
    "The figure of 86 percent of children suffering adverse reactions is "
    "true, but it is also misleading."
)


def test_dropped_record_excluded_from_output_and_logged():
    records = [
        {
            "id": 6419, "datasplit": "train",
            "url": "https://www.thequint.com/news/webqoof/x",
            "doc": _QUINT_6419_DOC,
        },
        {
            "id": 1, "datasplit": "train", "url": "https://www.reuters.com/y",
            "doc": "Clean evidence, unchanged.",
        },
    ]
    cleaned, difflog = clean_records(records)
    assert [r["id"] for r in cleaned] == [1]  # dropped record gone
    assert len(difflog) == 1
    entry = difflog[0]
    assert entry["id"] == 6419
    assert entry["datasplit"] == "train"
    assert entry["domain"] == "thequint.com"
    assert entry["dropped"] is True
    assert entry["reason"] == DROP_REASONS[("train", 6419)]


def test_drop_applies_before_stripping():
    # The doc matches thequint's claim_is rule ("...claim is false.") --
    # a dropped record must produce NO removal entries, only the drop
    # entry (drop happens before the stripper runs).
    records = [
        {
            "id": 6703, "datasplit": "train",
            "url": "https://www.thequint.com/news/webqoof/y",
            "doc": "What's the truth?: This claim is false.",
        },
    ]
    cleaned, difflog = clean_records(records)
    assert cleaned == []
    assert len(difflog) == 1
    assert difflog[0]["dropped"] is True
    assert "removals" not in difflog[0]
    assert "rules" not in difflog[0]


def test_same_id_on_other_split_is_not_dropped():
    # Keys are (datasplit, id): id 6419 exists on other splits too and
    # must survive there.
    records = [
        {
            "id": 6419, "datasplit": "val",
            "url": "https://www.reuters.com/y",
            "doc": "Clean evidence, unchanged.",
        },
    ]
    cleaned, difflog = clean_records(records)
    assert len(cleaned) == 1
    assert difflog == []


def test_drop_keys_contains_badly_scraped_thequint_records():
    # train:6419 (167 chars) and train:6703 (176 chars) -- junk fragments.
    assert ("train", 6419) in DROP_KEYS
    assert ("train", 6703) in DROP_KEYS
    # every key carries an auditable reason
    assert DROP_KEYS == frozenset(DROP_REASONS)
    assert all(DROP_REASONS[k] for k in DROP_KEYS)


# The 19 hybrid/dw_label_keys.txt records (all copies of the four
# "DW fact check:"-labelled compilations, 8533-19363 chars).
_DW_LABEL_KEYS = {
    ("val", 29), ("val", 1266), ("val", 1417),
    ("train", 1076), ("train", 1194), ("train", 1623), ("train", 2146),
    ("train", 2514), ("train", 3106), ("train", 3552), ("train", 4105),
    ("train", 5106), ("train", 5722), ("train", 6425), ("train", 7311),
    ("train", 7314), ("train", 8544), ("train", 9641),
    ("test", 188),
}

# Compilations found by the 2026-07-08 pristine-corpus sweep that the
# label-key list missed: the Crimea-bridge doc labels its 4 sub-claims
# "Verdict:" (not "DW fact check:"), and the 3-sub-claim oil-embargo doc
# never leaked labels into the hybrid output.
_DW_EXTRA_COMPILATION_KEYS = {
    ("train", 7214), ("train", 7462),  # Crimea bridge, 8522 chars
    ("train", 7325),                   # EU oil embargo, 5235 chars
}


def test_drop_keys_contains_dw_compilations():
    assert _DW_LABEL_KEYS <= DROP_KEYS
    assert _DW_EXTRA_COMPILATION_KEYS <= DROP_KEYS
    # The Rwanda doc (test:1997/test:2428, 5860 chars) is a SINGLE-claim
    # dw fact check, not a compilation -- it must be kept.
    assert ("test", 1997) not in DROP_KEYS
    assert ("test", 2428) not in DROP_KEYS
    # 2 thequint + 22 dw + 1 thin-evidence + 9 unlocatable-span; also
    # guards against silent dict-literal dupes.
    assert len(DROP_REASONS) == 34


# Records whose greedy_v3 production run (job 467303) carried a locator
# quote that could not be located verbatim (LLM quote drift: fused words
# like "wasBarely True", emoji substitution, paraphrase) -- residual
# verdict text likely remains, so they are dropped (user decision
# 2026-07-10).
_UNLOCATABLE_SPAN_KEYS = {
    ("train", 374), ("train", 1582), ("train", 1821), ("train", 3160),
    ("train", 4697), ("train", 4959), ("train", 6699), ("train", 8044),
    ("val", 2857),
}


def test_drop_keys_contains_thin_evidence_and_unlocatable_records():
    # newsmobile train:124: 1145 chars whose English content is claim
    # restatement + verdict; the remainder is untranslated boilerplate.
    assert ("train", 124) in DROP_KEYS
    assert _UNLOCATABLE_SPAN_KEYS <= DROP_KEYS
    assert all(DROP_REASONS[k] for k in _UNLOCATABLE_SPAN_KEYS)


def test_main_reports_drops_and_warns_on_unmatched_drop_keys(
    tmp_path, monkeypatch, capsys
):
    # One dropped record + one stripped record in a train file. Ids are
    # positional (assign_ids.py), so DROP_KEYS silently under-fires if the
    # pristine order changes: main() must warn about every train drop key
    # it did not match (e.g. train:6703), but NOT about val/test keys
    # (those splits were not processed here).
    records = [
        {
            "id": 6419, "datasplit": "train",
            "url": "https://www.thequint.com/news/webqoof/x",
            "doc": _QUINT_6419_DOC,
        },
        {
            "id": 0, "datasplit": "train",
            "url": "https://www.indiatoday.in/x",
            "doc": "Evidence here. JHOOTH BOLE KAUVA KAATE The number of "
                   "crows determines the intensity of the lie.",
        },
    ]
    src = tmp_path / "train_claims_quantemp.json"
    src.write_text(json.dumps(records), encoding="utf-8")
    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys, "argv",
        ["clean_splits", str(src), "--out-dir", str(out_dir)],
    )
    main()
    captured = capsys.readouterr()
    # drops are reported separately from strip changes
    assert "1 changed, 1 dropped" in captured.out
    assert "total changed: 1" in captured.out
    assert "dropped: 1" in captured.out
    # unmatched-key warning: train keys only
    assert "unmatched drop keys" in captured.err
    assert "train:6703" in captured.err
    assert "val:" not in captured.err
    assert "test:" not in captured.err
    # dropped record absent from the written split
    out = json.loads(
        (out_dir / "train_claims_quantemp.json").read_text(encoding="utf-8")
    )
    assert [r["id"] for r in out] == [0]


def test_dw_compilation_record_dropped():
    # Head of the real train:7325 doc (3 sub-claim compilation).
    records = [
        {
            "id": 7325, "datasplit": "train",
            "url": "https://www.dw.com/en/fact-check-is-india-violating-"
                   "the-eu-oil-embargo/a-6229107",
            "doc": 'Is India importing more oil from Russia?\n\nClaim: '
                   '" [...] India, taking advantage of the moment and '
                   'knowing about the sanctions, buys oil from Russia at a '
                   'discount of 40%," writes Twitter user Igor Fedorovsky '
                   'in Russian.\n\nDW fact check: True.',
        },
    ]
    cleaned, difflog = clean_records(records)
    assert cleaned == []
    assert len(difflog) == 1
    assert difflog[0]["dropped"] is True
    assert difflog[0]["domain"] == "dw.com"
    assert "compilation" in difflog[0]["reason"]
