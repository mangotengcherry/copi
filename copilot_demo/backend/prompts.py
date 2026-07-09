"""LLM system/user 프롬프트. LLM은 narration만 — 숫자/판정 생성 금지."""
import json

COPILOT_SYSTEM = (
    "당신은 반도체 공정 변경점의 EDS 통계 리뷰를 돕는 어시스턴트입니다. "
    "제공된 데이터에 없는 수치나 판정을 생성하지 말 것. "
    "공정 영향 여부를 단정하지 말 것. "
    "모든 주장에 근거 데이터 항목(group명, item_id, risk_level 등)을 언급할 것. "
    "한국어로 답변할 것. 반드시 제공된 JSON 스키마에 맞춰 응답할 것."
)

REPORT_SYSTEM = (
    "당신은 공정 변경점 EDS 리뷰 보고서 초안을 작성하는 어시스턴트입니다. "
    "제공된 데이터만 사용하고 새로운 수치나 판정을 만들지 말 것. "
    "간결한 한국어 markdown 보고서를 작성할 것."
)

COMMENT_SYSTEM = (
    "엔지니어의 자유 코멘트를 구조화하는 어시스턴트입니다. "
    "코멘트 내용만 사용하여 JSON으로 요약할 것. 한국어."
)


def build_copilot_user(context):
    return (
        "다음은 코드로 계산된 변경점 리뷰 컨텍스트입니다. 이 안의 값만 사용하세요.\n\n"
        + json.dumps(context, ensure_ascii=False, indent=2)
        + "\n\n위 데이터를 바탕으로 overview_summary(3~4문장), priority_groups, "
          "confounding_warnings, similar_case_insights, suggested_checks 를 작성하세요."
    )


def build_report_user(context):
    return (
        "다음 리뷰 컨텍스트로 markdown 보고서 초안을 작성하세요. 데이터 값만 사용:\n\n"
        + json.dumps(context, ensure_ascii=False, indent=2)
    )


def build_comment_user(comment):
    return (
        "다음 엔지니어 코멘트를 {\"summary\": str, \"tags\": [str], \"action_items\": [str]} "
        "형태 JSON으로 구조화하세요.\n코멘트: " + comment
    )
