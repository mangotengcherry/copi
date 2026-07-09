from pathlib import Path

SEED = 42

ITEM_GROUPS = ["WL", "Cell_Edge", "Peripheral", "Contact", "Metal", "Vth", "Leakage", "Speed"]
PROCESS_LEVEL1 = ["FE", "BE"]
PROCESS_MODULES = ["CVD", "Etch", "Photo", "Implant", "CMP", "Diffusion", "Metal"]
PROCESS_LAYERS = ["GateOx", "STI", "BitLine", "WordLine", "Contact", "M1", "M2", "Poly"]
CHANGE_TYPES = ["recipe_param", "hardware", "material", "route"]
FINAL_DECISIONS = ["공정영향", "설비영향", "Noise", "재검토"]
FEEDBACK_TAGS = ["중요", "Noise", "설비의심", "mix의심", "공정가능", "follow-up"]

SIGNIFICANCE_Q = 0.1

W_TREE = 0.40
W_SIGNATURE = 0.35
W_TYPE = 0.15
W_TEXT = 0.10

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "gemma3:12b"
OLLAMA_TEMPERATURE = 0.2
OLLAMA_TIMEOUT = 60

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# 정적 표준 체크리스트 (물량/통계/공정/최종 4섹션)
CHECKLIST = [
    {"section": "물량", "items": [
        "변경 적용 lot 수가 통계 판단에 충분한가",
        "before/after 기간의 물량 균형을 확인했는가",
    ]},
    {"section": "통계", "items": [
        "q_value < 0.1 유의 item을 모두 검토했는가",
        "effect_size가 실질적으로 유의미한 수준인가",
        "shift 방향이 물리적으로 설명 가능한가",
    ]},
    {"section": "공정", "items": [
        "변경 내용과 영향 group의 구조적 연관성을 확인했는가",
        "confounding(동시 변경/설비 편차)을 검토했는가",
        "유사 과거 사례의 결론과 비교했는가",
    ]},
    {"section": "최종", "items": [
        "최종 판정 근거를 데이터 항목으로 명시했는가",
        "follow-up action을 정의했는가",
    ]},
]
