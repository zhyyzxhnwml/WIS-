from __future__ import annotations

"""保序并发映射。

批量造数据是纯 I/O 等待，串行调用会让数万条样本耗时以天计。这里用线程池并发，
但保持输入顺序，避免打乱后续按序去重与分层采样。

单条失败（重试耗尽）不中断整批，只跳过并计数——长跑任务里一次网络抖动不该
让几小时的生成作废。
"""

from concurrent.futures import ThreadPoolExecutor


def pmap(fn, items, workers=1, label="pmap"):
    items = list(items)
    failures = []

    def safe(item):
        try:
            return fn(item)
        except Exception as exc:  # noqa: BLE001 — 单条失败降级为跳过
            failures.append(exc)
            return None

    if workers <= 1 or len(items) <= 1:
        results = [safe(item) for item in items]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(safe, items))

    if failures:
        print(f"[{label}] {len(failures)}/{len(items)} 条失败已跳过，首条: {failures[0]}")
    return results