from __future__ import annotations

import json

from .data import validate_response


def _extract_object(text):
    candidate = text.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(candidate[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None


def parse_prediction(text):
    if not isinstance(text, str) or not text.strip():
        return None, "prediction text is empty"
    obj = _extract_object(text)
    if obj is None:
        return None, "no parsable JSON object found in prediction"
    try:
        validate_response(obj)
    except ValueError as exc:
        return None, str(exc)
    return obj, None