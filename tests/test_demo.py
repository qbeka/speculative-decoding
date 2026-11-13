"""The replay page must open on a teachable recording and name every recording plainly."""

from importlib import resources

import pytest

from specdec.demo import block_view, pick_recordings, split_prompt, trace_label


class Tokenizer:
    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


def block(drafted, accepted, rejected_at, length=None):
    return {
        "block_length": length or drafted,
        "drafted": drafted,
        "accepted": accepted,
        "rejected_at": rejected_at,
        "delivered": accepted + 1,
        "draft_tokens": [97 + index for index in range(drafted)],
        "finished_at_s": 1.0,
        "proposal_s": 0.02,
        "verification_s": 0.04,
    }


def record(prompt_id, lengths, first=None):
    blocks = [block(length, length, None) for length in lengths]
    if first:
        blocks[0] = first
    return {"prompt_id": prompt_id, "blocks": blocks}


def test_a_prompt_recorded_twice_appears_once():
    records = [record("a", [1, 2, 3]), record("a", [1, 2]), record("b", [1])]
    picked = pick_recordings(records, lambda r: r["prompt_id"])
    assert [r["prompt_id"] for r in picked] == ["a", "b"]
    assert len(picked[0]["blocks"]) == 3


def test_two_prompts_with_one_label_appear_once():
    records = [record("a", [1, 2, 3]), record("b", [1, 2])]
    assert len(pick_recordings(records, lambda r: "same label")) == 1


def test_the_opening_recording_shows_a_rejection_in_its_first_draft():
    plain = record("plain", [1, 2, 3, 4])
    teaching = record("teaching", [5, 4], first=block(5, 1, 1))
    picked = pick_recordings([plain, teaching], lambda r: r["prompt_id"])
    assert picked[0]["prompt_id"] == "teaching"


def test_receipts_are_named_for_their_kind_and_merchant():
    example = {"family": "hotel", "truth": {"merchant": "Harbourview Inn"}, "id": "x-0001"}
    assert trace_label(example, "receipts_test") == "Hotel receipt, Harbourview Inn"


def test_general_text_keeps_the_whole_prompt_as_its_name():
    prompt = "How do antibiotics work, and why does resistance develop in the long run?"
    assert trace_label({"prompt": prompt}, "general_test") == prompt


def test_a_receipt_prompt_splits_into_its_task_and_receipt():
    task, receipt = split_prompt("Extract this. Reply with JSON.\n\nReceipt:\nGUEST FOLIO")
    assert (task, receipt) == ("Extract this.", "GUEST FOLIO")


def test_a_general_prompt_has_no_task_of_its_own():
    assert split_prompt("How do bees make honey?") == ("", "How do bees make honey?")


def test_a_block_keeps_its_draft_and_the_token_the_large_model_wrote():
    blocks = [block(3, 1, 1), block(2, 2, None)]
    tokens = [ord(letter) for letter in "aXabY"]
    view = block_view(Tokenizer(), {"blocks": blocks, "tokens": tokens})
    assert view[0]["draft"] == ["a", "b", "c"]
    assert view[0]["written"] == "X"
    assert view[1]["written"] == "Y"
    assert view[1]["rejected_at"] is None


@pytest.mark.parametrize("name", ["demo_template.html", "demo_foundation.css"])
def test_page_copy_has_no_long_dashes(name):
    text = resources.files("specdec").joinpath(name).read_text()
    assert "—" not in text
    assert "–" not in text


def test_the_tagline_lives_in_the_description_only():
    text = resources.files("specdec").joinpath("demo_template.html").read_text()
    assert '<meta name="description" content="Draft a few tokens and check them together">' in text
    assert "check them together</p>" not in text
