from __future__ import annotations

_BOOL_FIELDS = ("source_supported", "mapping_valid", "constraints_satisfied", "appropriate_deferral")


def blind_packet(predictions):
    return [
        {
            "prediction_id": item["prediction_id"],
            "question_id": item.get("question_id"),
            "completion": item["completion"],
        }
        for item in predictions
    ]


def summarize(predictions, reviews=None):
    reviews_by_id = {}
    if reviews is not None:
        for review in reviews:
            for field in _BOOL_FIELDS:
                if field in review and not isinstance(review[field], bool):
                    raise ValueError(f"{field} must be a boolean")
            reviews_by_id[review["prediction_id"]] = review

    by_scale = {}
    for prediction in predictions:
        key = str(prediction["scale"])
        report = by_scale.setdefault(key, {
            "total": 0, "reviewed": 0, "accepted": 0,
            "deferrals": 0, "appropriate_deferrals": 0, "questions": set(),
        })
        report["total"] += 1
        is_deferral = "status" in prediction["completion"]
        if is_deferral:
            report["deferrals"] += 1
        report["questions"].add(prediction.get("question_id"))
        review = reviews_by_id.get(prediction["prediction_id"])
        if review is not None:
            report["reviewed"] += 1
            if is_deferral:
                if review.get("appropriate_deferral"):
                    report["appropriate_deferrals"] += 1
            elif review.get("source_supported") and review.get("mapping_valid") and review.get("constraints_satisfied"):
                report["accepted"] += 1

    for report in by_scale.values():
        report["review_coverage"] = report["reviewed"] / report["total"] if report["total"] else None
        if reviews is None or report["reviewed"] == 0:
            report["accepted_among_reviewed"] = None
            report["accepted_mechanisms_per_question"] = None
        else:
            report["accepted_among_reviewed"] = report["accepted"] / report["reviewed"]
            questions = len(report["questions"])
            report["accepted_mechanisms_per_question"] = report["accepted"] / questions if questions else None
    return {"by_scale": by_scale}