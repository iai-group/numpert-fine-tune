"""Tests for the smoke-run input builder (offline)."""

import json

from src.number_annotation.smoke.run_smoke import build_smoke_input


def _data():
    """Build varied-length, varied-class records."""
    return [
        {
            "claim_id": f"r{i}",
            "taxonomy_label": "statistical" if i % 2 else "temporal",
            "label": "True" if i % 3 else "False",
            "doc": "x" * (i + 1),
        }
        for i in range(30)
    ]


class TestBuildSmokeInput:
    """Tests for build_smoke_input."""

    def test_writes_sample_of_n(self, tmp_path):
        """Verifies n records are sampled and written."""
        in_path = tmp_path / "in.json"
        out_path = tmp_path / "smoke_in.json"
        in_path.write_text(json.dumps(_data()), encoding="utf-8")

        count = build_smoke_input(
            str(in_path), str(out_path), n=9, seed=0
        )

        written = json.loads(out_path.read_text(encoding="utf-8"))
        assert count == 9
        assert len(written) == 9
