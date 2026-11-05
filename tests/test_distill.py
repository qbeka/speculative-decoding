import torch

from specdec.distill import WIDTH_BUCKET, make_batch, masked_mean


def row(length: int, prompt_length: int) -> dict:
    return {"tokens": list(range(1, length + 1)), "prompt_length": prompt_length}


def test_widths_round_up_to_a_bucket_so_shapes_repeat():
    short, _, _ = make_batch([row(WIDTH_BUCKET - 5, 3)], 0, "cpu")
    longer, _, _ = make_batch([row(WIDTH_BUCKET, 3)], 0, "cpu")
    assert short.shape == longer.shape == (1, WIDTH_BUCKET)


def test_trained_positions_predict_generated_tokens_only():
    input_ids, attention, trained = make_batch([row(5, 3)], 0, "cpu")
    assert attention[0].sum() == 5
    # Logits 2 and 3 predict tokens 3 and 4, the two generated positions.
    assert trained[0].nonzero().flatten().tolist() == [2, 3]
    assert input_ids[0, 5:].eq(0).all()


def test_masked_mean_ignores_positions_outside_the_mask():
    values = torch.tensor([1.0, 100.0, 3.0])
    mask = torch.tensor([True, False, True])
    assert masked_mean(values, mask) == 2.0
