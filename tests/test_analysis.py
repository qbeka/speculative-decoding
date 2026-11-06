"""The statistics must report what the records say."""

import gzip
import math

import pytest

from specdec.accuracy import first_json_object, score_output
from specdec.analysis import (
    arm_summary,
    bootstrap_ratio,
    compare,
    crossed_interval,
    fit_cost_prior,
    load_records,
    percentile,
    repeat_ratios,
    unit_rates,
)


def record(arm, chunk, repeat, tokens, wall, batch_size=1, workload="w", times=None, blocks=None):
    return {
        "arm": arm,
        "chunk": chunk,
        "repeat": repeat,
        "new_tokens": tokens,
        "wall_s": wall,
        "batch_size": batch_size,
        "workload": workload,
        "ttft_s": 0.1,
        "token_times_s": times or [],
        "blocks": blocks or [],
        "peak_rss_bytes": 2_000_000_000,
    }


def test_percentile_interpolates():
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5


def test_a_batch_counts_as_one_unit_timed_by_its_last_request():
    rates = unit_rates(
        [record("a", 0, 0, 10, 1.0, batch_size=2), record("a", 0, 0, 30, 2.0, batch_size=2)]
    )
    assert rates == {(0, 0): 20.0}


def test_rejected_draft_tokens_cannot_enter_the_rate():
    """The rate reads new_tokens, which comes from committed tokens only."""
    block = {"accepted": 1, "drafted": 8, "draft_passes": 8}
    summary = arm_summary([record("a", 0, 0, 2, 1.0, times=[0.1, 0.1], blocks=[block])])
    assert summary["tokens"] == 2
    assert summary["acceptance"] == pytest.approx(1 / 8)


def test_a_constant_ratio_has_a_tight_interval():
    log_ratios = {(unit, repeat): math.log(1.5) for unit in range(5) for repeat in range(3)}
    ratio, low, high = bootstrap_ratio(log_ratios)
    assert ratio == pytest.approx(1.5)
    assert low == pytest.approx(1.5)
    assert high == pytest.approx(1.5)


def test_noise_between_units_widens_the_interval_past_one():
    log_ratios = {(unit, 0): (0.5 if unit % 2 else -0.5) + 0.01 for unit in range(10)}
    _, low, high = bootstrap_ratio(log_ratios)
    assert low < 1 < high


def test_compare_pairs_only_matching_units():
    records = [
        record("a", 0, 0, 20, 1.0),
        record("b", 0, 0, 10, 1.0),
        record("a", 1, 0, 20, 1.0),  # No baseline for this unit.
    ]
    result = compare(records, "a", "b", "w", 1)
    assert result.ratio == pytest.approx(2.0)
    assert result.units == 1


def test_gaps_within_a_block_are_zero_and_are_reported():
    summary = arm_summary([record("a", 0, 0, 4, 1.0, times=[0.1, 0.1, 0.1, 0.5])])
    assert summary["median_gap_s"] == 0.0
    assert summary["p95_gap_s"] == pytest.approx(0.36)


def test_cost_prior_recovers_a_known_line():
    blocks = [
        {
            "draft_passes": 2,
            "proposal_s": 0.02,
            "verification_s": 0.02 + 0.001 * feed + 0.0001 * context,
            "sampling_s": 0.0,
            "target_feed": feed,
            "context_length": context,
        }
        for feed in (1, 3, 5, 9)
        for context in (50, 200, 400)
    ]
    prior = fit_cost_prior([{"blocks": blocks}])
    assert prior["draft_pass_s"] == pytest.approx(0.01)
    assert prior["verify_base_s"] == pytest.approx(0.02)
    assert prior["verify_per_token_s"] == pytest.approx(0.001)
    assert prior["verify_per_context_token_s"] == pytest.approx(0.0001)


def test_first_json_object_skips_text_and_bad_braces():
    assert first_json_object('Sure {not json} then {"a": 1} and {"b": 2}') == {"a": 1}
    assert first_json_object("no object") is None


def test_score_output_checks_each_field():
    truth = {
        "merchant": "Inkwell Books",
        "date": "2026-01-02",
        "currency": "CAD",
        "subtotal": 10.0,
        "tax": 0.5,
        "total": 10.5,
        "item_count": 2,
    }
    reply = (
        '{"merchant": "INKWELL BOOKS", "date": "2026-01-02", "currency": "USD", '
        '"subtotal": "10.00", "tax": 0.5, "total": 10.5, "item_count": 3}'
    )
    fields = score_output(reply, truth)["fields"]
    assert fields["merchant"] and fields["subtotal"]
    assert not fields["currency"]
    assert not fields["item_count"]
    assert score_output("no json here", truth)["parsed"] is False


def test_load_records_falls_back_to_the_compressed_copy(tmp_path):
    raw = tmp_path / "raw.jsonl"
    (tmp_path / "raw.jsonl.gz").write_bytes(gzip.compress(b'{"arm": "target"}\n'))
    assert load_records(raw) == [{"arm": "target"}]


def test_a_shock_to_one_whole_repeat_widens_only_the_crossed_interval():
    units, repeats = range(24), range(3)
    log_ratios = {
        (unit, repeat): (0.3 if repeat == 0 else 0.0) + 0.001 * unit
        for unit in units
        for repeat in repeats
    }
    _, low, high = bootstrap_ratio(log_ratios)
    crossed_low, crossed_high = crossed_interval(log_ratios)
    assert crossed_high / crossed_low > 1.2 * high / low


def test_repeat_ratios_are_in_repeat_order():
    log_ratios = {(0, 0): math.log(2.0), (1, 0): math.log(2.0), (0, 1): 0.0, (1, 1): 0.0}
    assert repeat_ratios(log_ratios) == pytest.approx([2.0, 1.0])
