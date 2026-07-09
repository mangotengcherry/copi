"""유사 변경점 검색: (1) hard filter (2) 4축 가중합. 설명가능성 위해 breakdown 노출."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend import data_store as ds
from backend.stats_summary import significant_signature
from backend.constants import W_TREE, W_SIGNATURE, W_TYPE, W_TEXT
from backend.models import SimilarCase, SimilarBreakdown, SimilarResponse


def tree_proximity(a, b):
    if a["level1"] == b["level1"] and a["level2"] == b["level2"] and a["level3"] == b["level3"]:
        return 1.0
    if a["level1"] == b["level1"] and a["level2"] == b["level2"]:
        return 0.6
    if a["level1"] == b["level1"]:
        return 0.3
    return 0.0


def signature_similarity(sig_a, sig_b):
    """유의 group set의 Jaccard + 교집합에서 방향 일치 보너스."""
    ga, gb = set(sig_a), set(sig_b)
    if not ga or not gb:
        return 0.0
    inter = ga & gb
    union = ga | gb
    jaccard = len(inter) / len(union)
    if not inter:
        return round(jaccard, 4)
    agree = sum(1 for g in inter if sig_a[g] == sig_b[g]) / len(inter)
    return round(min(1.0, 0.7 * jaccard + 0.3 * jaccard * agree + 0.3 * agree * (len(inter) / len(union))), 4)


def text_similarity(text_a, text_b, corpus):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3))
    vec.fit(corpus + [text_a, text_b])
    m = vec.transform([text_a, text_b])
    return float(cosine_similarity(m[0], m[1])[0][0])


def _hard_filter(base, cand):
    return (base["process_step"]["level1"] == cand["process_step"]["level1"]
            or base["change_type"] == cand["change_type"])


def find_similar(cid, top_k=3):
    base = ds.get_change(cid)
    base_sig = significant_signature(cid)
    candidates = [c for c in ds.get_changes()
                  if c["status"] == "past" and c["change_id"] != cid]
    corpus = [c["description_ko"] for c in ds.get_changes()]

    scored = []
    for cand in candidates:
        if not _hard_filter(base, cand):
            continue
        tree = tree_proximity(base["process_step"], cand["process_step"])
        sig = signature_similarity(base_sig, significant_signature(cand["change_id"]))
        typ = 1.0 if base["change_type"] == cand["change_type"] else 0.0
        txt = text_similarity(base["description_ko"], cand["description_ko"], corpus)
        total = W_TREE * tree + W_SIGNATURE * sig + W_TYPE * typ + W_TEXT * txt
        card = ds.get_review_card(cand["change_id"])
        scored.append(SimilarCase(
            change_id=cand["change_id"],
            description_ko=cand["description_ko"],
            change_type=cand["change_type"],
            process_path=cand["process_step"]["path"],
            final_decision=card["final_decision"] if card else "N/A",
            total_score=round(total, 4),
            breakdown=SimilarBreakdown(
                tree=round(tree, 4), signature=round(sig, 4),
                type=round(typ, 4), text=round(txt, 4)),
        ))
    scored.sort(key=lambda s: s.total_score, reverse=True)
    return SimilarResponse(candidates=scored[:top_k])
