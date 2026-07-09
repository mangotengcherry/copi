from backend.models import Change, ProcessStep, CopilotSummary, StatResult


def test_change_roundtrip():
    c = Change(
        change_id="CHG-2024-001",
        process_step=ProcessStep(level1="FE", level2="CVD", level3="GateOx", path="FE/CVD/GateOx"),
        change_type="recipe_param", change_direction="증착온도 +5%",
        description_ko="증착 온도를 5% 상향 조정하였다.",
        period_start="2024-03-01", period_end="2024-03-20", lot_count=40, status="current",
    )
    assert c.model_dump()["process_step"]["path"] == "FE/CVD/GateOx"


def test_copilot_summary_schema_has_required_keys():
    schema = CopilotSummary.model_json_schema()
    for k in ["overview_summary", "priority_groups", "confounding_warnings",
              "similar_case_insights", "suggested_checks"]:
        assert k in schema["properties"]


def test_statresult_optional_hist():
    s = StatResult(change_id="CHG-2024-001", item_id="EDS_001", item_group="WL",
                   p_value=0.001, q_value=0.01, effect_size=0.8, shift_direction="up")
    assert s.mean_before is None
