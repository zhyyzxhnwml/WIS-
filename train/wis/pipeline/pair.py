from __future__ import annotations

"""阶段 3：远域配对。

用字符 n-gram TF-IDF（对中文有效、无需模型下载）算问题↔机制的表面距离，
每个问题挑 k 个「表面远、领域不同」的机制，距离 < 阈值的近域丢弃。
"""

import numpy as np

from .io import stable_id


def _problem_text(problem):
    return " ".join(
        [problem.get("question", ""), problem.get("function", ""), problem.get("behavior", ""), problem.get("structure", "")]
    )


def _mechanism_text(mechanism):
    return " ".join(
        [mechanism.get("mechanism", ""), mechanism.get("mechanism_description", ""), mechanism.get("function", "")]
    )


def mechanism_group(mechanism):
    return stable_id("mg", mechanism["mechanism"], mechanism["source_domain"])


def _embed(texts):
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    matrix = vectorizer.fit_transform(texts)
    norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A1
    norms[norms == 0] = 1.0
    return matrix, norms


def _lexical_distance(problems, mechanisms):
    problem_texts = [_problem_text(p) for p in problems]
    mechanism_texts = [_mechanism_text(m) for m in mechanisms]
    matrix, norms = _embed(problem_texts + mechanism_texts)
    problem_mat = matrix[: len(problems)]
    mechanism_mat = matrix[len(problems):]
    problem_norms = norms[: len(problems)]
    mechanism_norms = norms[len(problems):]
    similarity = (problem_mat @ mechanism_mat.T).toarray()
    similarity = similarity / np.outer(problem_norms, mechanism_norms)
    return 1.0 - np.clip(similarity, 0.0, 1.0)


def pair_far_domains(problems, mechanisms, k=6, min_distance=0.6, cross_domain=True, domain_distance=None):
    """远域配对。

    传入 domain_distance（{「A|B」: 距离}）时用它打分：字符 n-gram 在长文本间恒趋近 1，
    没有区分度；领域质心距离才有。未传时退回词面距离。
    """
    if not problems or not mechanisms:
        return []
    lexical = None if domain_distance is not None else _lexical_distance(problems, mechanisms)

    # 全局用量均衡：每个问题优先挑「用得最少」的源领域，其次取距离最远的。
    # 早前按领域名字典序轮询，排序靠后的领域（计算机、航空航天）会被系统性挤出，
    # 计算机因 unicode 排序最末直接归零，九领域覆盖熵随之崩塌。
    usage = {}
    pairs = []
    for i, problem in enumerate(problems):
        target_domain = problem.get("source_domain")
        if domain_distance is not None and target_domain:
            scores = np.array(
                [float(domain_distance.get(f"{target_domain}|{m['source_domain']}", 0.0)) for m in mechanisms]
            )
        else:
            scores = lexical[i]
        order = np.argsort(-scores)
        best = {}
        for j in order:
            mechanism = mechanisms[int(j)]
            score = float(scores[j])
            if score < min_distance:
                break
            domain = mechanism["source_domain"]
            if cross_domain and target_domain and domain == target_domain:
                continue
            if domain not in best:
                best[domain] = (score, int(j))
        domains = sorted(best, key=lambda d: (usage.get(d, 0), -best[d][0]))
        for domain in domains[:k]:
            dist, j = best[domain]
            usage[domain] = usage.get(domain, 0) + 1
            mechanism = mechanisms[j]
            pairs.append(
                {
                    "pair_id": stable_id("pair", problem["problem_id"], mechanism["raw_id"]),
                    "problem_id": problem["problem_id"],
                    "problem_group": problem["problem_group"],
                    "target_domain": target_domain,
                    "raw_id": mechanism["raw_id"],
                    "mechanism_group": mechanism_group(mechanism),
                    "source_domain": mechanism["source_domain"],
                    "distance": round(dist, 4),
                }
            )
    return pairs