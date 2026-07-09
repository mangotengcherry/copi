"""Ollama 호출 wrapper: format=json + schema, 1회 재시도, rule-based fallback."""
import json
import requests

from backend.constants import OLLAMA_URL, OLLAMA_MODEL, OLLAMA_TEMPERATURE, OLLAMA_TIMEOUT
from backend.models import CopilotSummary, PriorityGroup, SimilarCaseInsight
from backend import prompts


def call_ollama(system, user, schema, timeout=OLLAMA_TIMEOUT):
    """Ollama /api/chat 호출. 실패/파싱오류 시 예외 발생."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "format": schema,
        "options": {"temperature": OLLAMA_TEMPERATURE},
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    resp.raise_for_status()
    content = resp.json()["message"]["content"]
    return json.loads(content)


def fallback_summary(context):
    ov = context["overview"]
    ranking = context.get("ranking", [])
    similar = context.get("similar", [])
    high = [r for r in ranking if r["risk_level"] in ("High", "Med")]
    groups_txt = ", ".join(f"{r['group']}({r['risk_level']})" for r in high) or "없음"
    summary = (
        f"{ov['change_direction']} 변경({ov['process_path']})에 대해 총 "
        f"{ov['significant_item_count']}개 유의 item이 관찰되었습니다. "
        f"위험도 상위 group은 {groups_txt} 입니다. "
        f"수치는 코드 계산 결과이며 공정 영향 여부는 엔지니어 판단이 필요합니다."
    )
    return CopilotSummary(
        overview_summary=summary,
        priority_groups=[PriorityGroup(
            group=r["group"],
            reason=f"risk_level {r['risk_level']} (score {r['risk_score']}), 대표 item {r['representative_item']}")
            for r in high[:3]],
        confounding_warnings=["설비 편차·동시 변경 등 confounding 요인을 별도 검토 필요."],
        similar_case_insights=[SimilarCaseInsight(
            change_id=s["change_id"],
            relevance=f"유사도 {s['total_score']}, 과거 판정 {s.get('final_decision','N/A')}")
            for s in similar[:3]],
        suggested_checks=[
            "q_value < 0.1 유의 item 목록 재확인",
            "위험 group 대표 item의 before/after 분포 확인",
            "유사 과거 사례의 최종 판정과 비교",
        ],
    )


def generate_copilot_summary(context):
    """(CopilotSummary, used_llm: bool). 파싱 실패 시 1회 재시도 후 fallback."""
    schema = CopilotSummary.model_json_schema()
    for attempt in range(2):
        try:
            raw = call_ollama(prompts.COPILOT_SYSTEM, prompts.build_copilot_user(context), schema)
            return CopilotSummary.model_validate(raw), True
        except Exception:
            continue
    return fallback_summary(context), False


def _fallback_report(context):
    ov = context["overview"]
    ranking = context.get("ranking", [])
    similar = context.get("similar", [])
    lines = [
        f"# 공정 변경점 EDS 리뷰 보고서 (초안)",
        "",
        f"## 변경 개요",
        f"- 변경 ID: {ov['change_id']}",
        f"- 변경 내용: {ov['change_direction']} ({ov['process_path']})",
        f"- 유의 item 수: {ov['significant_item_count']} / 200",
        "",
        f"## 위험 group",
    ]
    for r in ranking:
        lines.append(f"- {r['group']}: {r['risk_level']} (score {r['risk_score']}, 대표 {r['representative_item']})")
    lines += ["", "## 유사 과거 사례"]
    for s in similar:
        lines.append(f"- {s['change_id']}: 유사도 {s['total_score']}, 판정 {s.get('final_decision','N/A')}")
    lines += ["", "## 검토 의견", "- (엔지니어 작성 필요)"]
    return "\n".join(lines)


def generate_report(context):
    """(markdown, used_llm: bool)."""
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": prompts.REPORT_SYSTEM},
                {"role": "user", "content": prompts.build_report_user(context)},
            ],
            "stream": False,
            "options": {"temperature": OLLAMA_TEMPERATURE},
        }
        resp = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        resp.raise_for_status()
        return resp.json()["message"]["content"], True
    except Exception:
        return _fallback_report(context), False


def structure_comment(comment):
    """자유 코멘트 구조화. 실패 시 원문 반환."""
    if not comment.strip():
        return {"summary": "", "tags": [], "action_items": []}
    schema = {"type": "object", "properties": {
        "summary": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "action_items": {"type": "array", "items": {"type": "string"}}},
        "required": ["summary", "tags", "action_items"]}
    try:
        return call_ollama(prompts.COMMENT_SYSTEM, prompts.build_comment_user(comment), schema)
    except Exception:
        return {"summary": comment, "tags": [], "action_items": []}
