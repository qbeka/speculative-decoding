import json
from dataclasses import dataclass

import pytest

from specdec.report import find_comparison, law_audit_range


@dataclass
class Item:
    arm: str
    baseline: str
    workload: str
    batch_size: int


def test_find_comparison_refuses_a_pair_that_was_not_frozen():
    items = [Item("adaptive/135m", "target", "receipts_test", 1)]
    assert find_comparison(items, "adaptive/135m", "target", "receipts_test", 1) is items[0]
    with pytest.raises(KeyError):
        find_comparison(items, "adaptive/135m", "target", "receipts_test", 4)


def test_law_audit_range_leaves_out_the_negative_control(tmp_path):
    path = tmp_path / "audit.json"
    rows = [
        {"check": "fixed_k2", "p_value": 0.2},
        {"check": "adaptive", "p_value": 0.6},
        {"check": "negative_control_draft_only", "p_value": 0.0},
    ]
    path.write_text(json.dumps({"draws_per_arm": 1500, "rows": rows}))
    assert law_audit_range(path) == (1500, 0.2, 0.6)
