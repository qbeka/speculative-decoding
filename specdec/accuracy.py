"""Task accuracy on receipts. This says how good the text is, not whether the sampler is right.

The sampler's correctness is settled by the exact-law tests. Accuracy is reported
separately because speculation must not change it, so a large gap between arms
would point to a bug.
"""

import json

FIELDS = ("merchant", "date", "currency", "subtotal", "tax", "total", "item_count")
NUMERIC_FIELDS = {"subtotal", "tax", "total", "item_count"}
AMOUNT_TOLERANCE = 0.005


def first_json_object(text: str) -> dict | None:
    """Return the first parseable JSON object in text, or None."""
    decoder = json.JSONDecoder()
    position = text.find("{")
    while position != -1:
        try:
            value, _ = decoder.raw_decode(text, position)
        except json.JSONDecodeError:
            position = text.find("{", position + 1)
            continue
        if isinstance(value, dict):
            return value
        position = text.find("{", position + 1)
    return None


def field_matches(field: str, predicted, truth) -> bool:
    if field in NUMERIC_FIELDS:
        try:
            return abs(float(predicted) - float(truth)) <= AMOUNT_TOLERANCE
        except (TypeError, ValueError):
            return False
    return str(predicted).strip().casefold() == str(truth).strip().casefold()


def score_output(text: str, truth: dict) -> dict:
    """Return which fields are right. A reply with no JSON gets every field wrong."""
    predicted = first_json_object(text)
    if predicted is None:
        return {"parsed": False, "fields": {field: False for field in FIELDS}}
    return {
        "parsed": True,
        "fields": {
            field: field in predicted and field_matches(field, predicted[field], truth[field])
            for field in FIELDS
        },
    }
