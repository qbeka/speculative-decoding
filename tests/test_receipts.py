"""The receipt data must be reproducible and its splits must not leak."""

from specdec.receipts import FAMILIES, SPLITS, build_examples


def test_every_family_is_in_exactly_one_split():
    assigned = [family for families in SPLITS.values() for family in families]
    assert sorted(assigned) == sorted(FAMILIES)
    assert len(assigned) == len(set(assigned))


def test_examples_are_deterministic():
    assert build_examples("test", 3, seed=7) == build_examples("test", 3, seed=7)


def test_evaluation_merchants_never_appear_in_training():
    train = " ".join(example["prompt"] for example in build_examples("train", 40, seed=1))
    for split in ("validation", "test"):
        for example in build_examples(split, 40, seed=1):
            assert example["truth"]["merchant"] not in train


def test_truth_amounts_add_up():
    for split in SPLITS:
        for example in build_examples(split, 25, seed=3):
            truth = example["truth"]
            assert (
                truth["total"] >= truth["subtotal"] + truth["tax"] - 0.02 or truth["subtotal"] < 0
            )


def test_true_values_are_visible_in_the_prompt():
    for example in build_examples("test", 10, seed=2):
        assert example["truth"]["merchant"].upper() in example["prompt"].upper()
