import json
import csv
import sys
from code.metrics import dataset_metrics

def test_count_labels_and_generate_csv_basic(tmp_path):
    data = [
        {"label": "True"},
        {"label": "False"},
        {"label": "True"},
        {"label": "Other"},
        {"no_label": "x"},
    ]
    json_path = tmp_path / "data.json"
    csv_path = tmp_path / "stats.csv"
    with open(json_path, "w") as f:
        json.dump(data, f)

    result = dataset_metrics.count_labels_and_generate_csv(str(json_path), str(csv_path))

    assert result == {"True": 2, "False": 1, "Total": 5}
    assert csv_path.exists()

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    metrics = {row["Metric"]: int(row["Count"]) for row in rows}
    assert metrics == {"True Labels": 2, "False Labels": 1, "Total Elements": 5}


def test_count_labels_ignores_non_string_labels(tmp_path):
    data = [
        {"label": True},
        {"label": "true"},
        {"label": "FALSE"},
        {"label": None},
        {},
        {"label": 0},
        {"label": "False"},
        {"label": "True"},
    ]
    json_path = tmp_path / "data.json"
    csv_path = tmp_path / "stats.csv"
    with open(json_path, "w") as f:
        json.dump(data, f)

    result = dataset_metrics.count_labels_and_generate_csv(str(json_path), str(csv_path))

    assert result == {"True": 1, "False": 1, "Total": 8}
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    metrics = {row["Metric"]: int(row["Count"]) for row in rows}
    assert metrics == {"True Labels": 1, "False Labels": 1, "Total Elements": 8}


def test_count_labels_error_on_missing_file(tmp_path, capsys):
    json_path = tmp_path / "missing.json"  # does not exist
    csv_path = tmp_path / "stats.csv"

    result = dataset_metrics.count_labels_and_generate_csv(str(json_path), str(csv_path))

    assert "error" in result and result["error"]
    assert not csv_path.exists()
    out = capsys.readouterr().out
    assert "Error:" in out


def test_main_cli_writes_csv_and_prints_result(tmp_path, monkeypatch, capsys):
    data = [
        {"label": "True"},
        {"label": "False"},
        {"label": "True"},
    ]
    json_path = tmp_path / "data.json"
    csv_path = tmp_path / "stats.csv"
    with open(json_path, "w") as f:
        json.dump(data, f)

    monkeypatch.setattr(sys, "argv", ["prog", str(json_path), str(csv_path)])
    dataset_metrics.main()

    captured = capsys.readouterr().out
    assert "Statistics saved to" in captured
    assert "Result:" in captured
    assert "'True': 2" in captured
    assert "'False': 1" in captured
    assert "'Total': 3" in captured

    assert csv_path.exists()
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    metrics = {row["Metric"]: int(row["Count"]) for row in rows}
    assert metrics == {"True Labels": 2, "False Labels": 1, "Total Elements": 3}