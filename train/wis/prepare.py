from __future__ import annotations


def grouped_split(rows, ratios=(0.8, 0.2)):
    groups = {}
    for row in rows:
        groups.setdefault(row["problem_group"], []).append(row)
    keys = sorted(groups)
    total = len(keys)
    if total == 0:
        return {}
    n_train = max(1, int(round(total * ratios[0]))) if total > 1 else total
    train_keys = set(keys[:n_train])
    result = {"train": [], "val": []}
    for key in keys:
        target = "train" if key in train_keys else "val"
        result[target].extend(groups[key])
    return {name: items for name, items in result.items() if items}