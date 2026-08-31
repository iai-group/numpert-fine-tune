"""Tests for stratified seed sampling."""

from src.number_annotation.smoke.sample_seed import (
    smoke_sample,
    stratified_sample,
)


def _make_data():
    """Build a small dataset with two taxonomy x label strata."""
    data = []
    for i in range(40):
        data.append(
            {
                "claim_id": f"a{i}",
                "taxonomy_label": "statistical",
                "label": "True",
            }
        )
    for i in range(60):
        data.append(
            {
                "claim_id": f"b{i}",
                "taxonomy_label": "temporal",
                "label": "False",
            }
        )
    return data


class TestStratifiedSample:
    """Tests for stratified_sample."""

    def test_sample_size(self):
        """Verifies the requested sample size is honored."""
        out = stratified_sample(_make_data(), n=20, seed=0)
        assert len(out) == 20

    def test_strata_proportional(self):
        """Verifies strata are sampled roughly proportionally."""
        out = stratified_sample(_make_data(), n=20, seed=0)
        stat = sum(1 for r in out if r["taxonomy_label"] == "statistical")
        temp = sum(1 for r in out if r["taxonomy_label"] == "temporal")
        # 40:60 split over 20 -> about 8:12.
        assert stat == 8
        assert temp == 12

    def test_reproducible(self):
        """Verifies the same seed yields the same sample."""
        a = stratified_sample(_make_data(), n=20, seed=7)
        b = stratified_sample(_make_data(), n=20, seed=7)
        assert [r["claim_id"] for r in a] == [r["claim_id"] for r in b]

    def test_n_larger_than_data_returns_all(self):
        """Verifies requesting more than available returns all rows."""
        data = _make_data()
        out = stratified_sample(data, n=1000, seed=0)
        assert len(out) == len(data)


def _len_data():
    """Build data with varying doc lengths and classes."""
    data = []
    for i in range(30):
        data.append(
            {
                "claim_id": f"r{i}",
                "taxonomy_label": "statistical" if i % 2 else "temporal",
                "label": "True" if i % 3 else "False",
                "doc": "x" * (i + 1),
            }
        )
    return data


class TestSmokeSample:
    """Tests for smoke_sample (length + class variety)."""

    def test_size(self):
        """Verifies the smoke sample size."""
        assert len(smoke_sample(_len_data(), n=9, seed=0)) == 9

    def test_spans_length_buckets(self):
        """Verifies short and long docs are both represented."""
        out = smoke_sample(_len_data(), n=9, seed=0)
        lengths = sorted(len(r["doc"]) for r in out)
        assert lengths[0] <= 10
        assert lengths[-1] >= 20

    def test_reproducible(self):
        """Verifies the same seed yields the same sample."""
        a = smoke_sample(_len_data(), 9, 7)
        b = smoke_sample(_len_data(), 9, 7)
        assert [r["claim_id"] for r in a] == [r["claim_id"] for r in b]
