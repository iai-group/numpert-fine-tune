"""Test suite for the remove_referance module in raw_dataset_preprocessing."""

import json
from pathlib import Path
import pytest


from code.data_preprocessing.raw_dataset_preprocessing.remove_referance import (
    remove_gold_verdict,
    find_json_files,
    process_json_file,
    process_folder,
)


def read_text(p: Path) -> str:
    """Read text from a file."""
    return p.read_text(encoding="utf-8")


def write_text(p: Path, text: str) -> None:
    """Write text to a file."""
    p.write_text(text, encoding="utf-8")


def write_json(p: Path, data) -> None:
    """Write data to a JSON file."""
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def test_remove_gold_verdict_basic_behavior():
    """Test basic functionality of remove_gold_verdict."""
    text = "This is a fact. It is correct. Our verdict is true. After."
    assert remove_gold_verdict(text) == "This is a fact."


def test_remove_gold_verdict_empty_and_none():
    """Test behavior with empty string and None."""
    assert remove_gold_verdict("") == ""
    assert remove_gold_verdict(None) is None  # type: ignore[arg-type]


def test_remove_gold_verdict_filters_and_last_sentence():
    """Test filtering and last sentence removal."""
    text = "Alpha is misleading. Gamma."
    # Remove last sentence "Gamma.", filter out "misleading"
    assert remove_gold_verdict(text) == "Gamma."


def test_find_json_files_recursively(tmp_path: Path):
    """Test finding JSON files recursively in a directory."""
    # Create nested structure
    a = tmp_path / "a.json"
    bdir = tmp_path / "sub" / "dir"
    bdir.mkdir(parents=True)
    b = bdir / "b.json"
    c = tmp_path / "c.txt"
    write_json(a, [])
    write_json(b, [])
    write_text(c, "not json")
    found = find_json_files(tmp_path)
    assert set(map(lambda p: p.resolve(), found)) == {a.resolve(), b.resolve()}


def test_process_json_file_modifies_and_writes(
    tmp_path: Path, capsys: pytest.CaptureFixture
):
    """Test that process_json_file modifies 'doc' field and writes changes."""
    fp = tmp_path / "data.json"
    data = [
        {"doc": "First. Verdict false."},  # will be modified to "First."
        {"doc": ""},  # unchanged
        {"doc": 123},  # non-string
        {},  # missing doc
    ]
    write_json(fp, data)

    process_json_file(fp)

    out = capsys.readouterr().out
    assert "Processed 'doc' in" in out
    assert "No 'doc' field found or 'doc' is not a string" in out
    assert f"Updated file: {fp}" in out

    with fp.open("r", encoding="utf-8") as f:
        updated = json.load(f)
    assert updated[0]["doc"] == "First."
    assert updated[1]["doc"] == ""


def test_process_json_file_dry_run_no_write(
    tmp_path: Path, capsys: pytest.CaptureFixture
):
    """Test that process_json_file does not write changes in dry run mode."""
    fp = tmp_path / "dry.json"
    data = [{"doc": "A. true. B."}]  # will become "A."
    write_json(fp, data)

    process_json_file(fp, dry_run=True)
    out = capsys.readouterr().out
    assert "Dry run enabled. Changes not saved for file" in out

    with fp.open("r", encoding="utf-8") as f:
        unchanged = json.load(f)
    assert unchanged == data


def test_process_json_file_invalid_json(
    tmp_path: Path, capsys: pytest.CaptureFixture
):
    """Test that process_json_file handles invalid JSON gracefully."""
    fp = tmp_path / "bad.json"
    write_text(fp, "{ invalid json")

    process_json_file(fp)
    out = capsys.readouterr().out
    assert f"Error: The file '{fp}' does not contain valid JSON." in out


def test_process_json_file_nonlist_root(
    tmp_path: Path, capsys: pytest.CaptureFixture
):
    """Test that process_json_file skips files with non-list root."""
    fp = tmp_path / "obj.json"
    write_json(fp, {"doc": "A. B."})
    process_json_file(fp)
    out = capsys.readouterr().out
    assert f"Skipping {fp}: JSON data is not a list." in out


def test_process_json_file_no_changes_made(
    tmp_path: Path, capsys: pytest.CaptureFixture
):
    """Test reports no changes when nothing is modified."""
    fp = tmp_path / "nochange.json"
    # Empty doc returns empty unchanged (modified flag stays False)
    write_json(fp, [{"doc": ""}])
    process_json_file(fp)
    out = capsys.readouterr().out
    assert f"No changes made to file: {fp}" in out
    # Content should be identical
    with fp.open("r", encoding="utf-8") as f:
        assert json.load(f) == [{"doc": ""}]


def test_process_folder_overwrite_original(
    tmp_path: Path, capsys: pytest.CaptureFixture
):
    """Test that modifies files in place without output directory."""
    input_dir = tmp_path / "in2"
    input_dir.mkdir(parents=True)
    f = input_dir / "x.json"
    write_json(f, [{"doc": "First. false."}])  # will become "First."

    process_folder(input_dir, output_dir=None, dry_run=False)
    out = capsys.readouterr().out

    assert f"Found 1 JSON file(s) in '{input_dir}'." in out
    assert f"Updated file: {f}" in out

    with f.open("r", encoding="utf-8") as fh:
        updated = json.load(fh)
    assert updated == [{"doc": "First."}]


def test_process_folder_missing_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture
):
    """Test that process_folder handles missing input directory."""
    missing = tmp_path / "does-not-exist"
    process_folder(missing)
    out = capsys.readouterr().out
    assert f"Error: The input directory '{missing}' does not exist" in out


def test_process_folder_no_json_files(
    tmp_path: Path, capsys: pytest.CaptureFixture
):
    """Test that process_folder handles directories with no JSON files."""
    d = tmp_path / "empty"
    d.mkdir(parents=True)
    # create non-json file
    write_text(d / "note.txt", "hello")
    process_folder(d)
    out = capsys.readouterr().out
    assert f"No JSON files found in the directory '{d}'." in out


def test_process_folder_with_output_dir_nonlist_json(
    tmp_path: Path, capsys: pytest.CaptureFixture
):
    """Test process_folder with output directory and non-list JSON root."""
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir(parents=True)

    f = input_dir / "obj.json"
    write_json(f, {"doc": "not a list"})

    process_folder(input_dir, output_dir=output_dir, dry_run=False)
    out = capsys.readouterr().out

    assert f"Skipping {f}: JSON data is not a list." in out
    # No output file should be created for skipped files
    out_file = output_dir / "obj.json"
    assert not out_file.exists()


def test_process_folder_with_output_dir_no_changes(
    tmp_path: Path, capsys: pytest.CaptureFixture
):
    """Test process_folder with output directory and no changes made."""
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir(parents=True)

    f = input_dir / "unchanged.json"
    write_json(f, [{"doc": "Simple text without verdict words."}])

    process_folder(input_dir, output_dir=output_dir, dry_run=False)
    out = capsys.readouterr().out

    assert f"No changes made to file: {f}" in out
    # File should still be copied to output even if unchanged
    out_file = output_dir / "unchanged.json"
    assert not out_file.exists()  # No file created when no changes


def test_process_folder_with_output_dir_missing_doc_field(
    tmp_path: Path, capsys: pytest.CaptureFixture
):
    """Test process_folder with output directory and missing 'doc' field."""
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir(parents=True)

    f = input_dir / "nodoc.json"
    write_json(f, [{"other": "field"}, {"doc": 123}])

    process_folder(input_dir, output_dir=output_dir, dry_run=False)
    out = capsys.readouterr().out

    assert "No 'doc' field found or 'doc' is not a string" in out
    # No changes means no output file created
    out_file = output_dir / "nodoc.json"
    assert not out_file.exists()


def test_process_folder_with_output_dir_exception_handling(
    tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch
):
    """Test process_folder with output directory and exception handling."""
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir(parents=True)

    f = input_dir / "test.json"
    write_json(f, [{"doc": "Test. true."}])

    # Mock json.dump to raise an exception
    def mock_json_dump(*args, **kwargs):
        """Mock json.dump to raise an IOError."""
        raise IOError("Simulated write error")

    monkeypatch.setattr("json.dump", mock_json_dump)

    process_folder(input_dir, output_dir=output_dir, dry_run=False)
    out = capsys.readouterr().out

    assert "An unexpected error occurred while processing" in out
    assert "Simulated write error" in out
