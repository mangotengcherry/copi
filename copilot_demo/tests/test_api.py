from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def _current_id():
    changes = client.get("/changes").json()
    return [c for c in changes if c["status"] == "current"][0]["change_id"]


def test_list_changes():
    r = client.get("/changes")
    assert r.status_code == 200 and len(r.json()) == 15


def test_overview_ranking_similar():
    cid = _current_id()
    assert client.get(f"/changes/{cid}/overview").json()["total_items_tested"] == 200
    assert len(client.get(f"/changes/{cid}/ranking").json()) == 8
    assert "candidates" in client.get(f"/changes/{cid}/similar").json()


def test_checklist_has_four_sections():
    r = client.get("/checklist").json()
    assert [s["section"] for s in r["sections"]] == ["물량", "통계", "공정", "최종"]


def test_copilot_summary_returns_schema_even_without_llm():
    cid = _current_id()
    r = client.post(f"/changes/{cid}/copilot_summary")
    assert r.status_code == 200
    body = r.json()
    for k in ["overview_summary", "priority_groups", "confounding_warnings",
              "similar_case_insights", "suggested_checks"]:
        assert k in body


def test_feedback_and_report():
    cid = _current_id()
    fb = client.post(f"/changes/{cid}/feedback", json={
        "checklist_state": {"물량-0": True}, "group_feedback": [{"group": "Metal", "tags": ["중요"]}],
        "final_decision": "재검토", "comment": "추가 모니터링 필요"})
    assert fb.status_code == 200 and fb.json()["saved"] is True
    rep = client.post(f"/changes/{cid}/report_draft")
    assert rep.status_code == 200 and rep.json()["markdown"].startswith("#")


def test_histogram_endpoint():
    cid = _current_id()
    ranking = client.get(f"/changes/{cid}/ranking").json()
    item = ranking[0]["representative_item"]
    h = client.get(f"/changes/{cid}/histogram/{item}").json()
    assert len(h["before"]) > 0 and len(h["after"]) > 0


def test_recommendations_endpoint():
    cid = _current_id()
    r = client.get(f"/changes/{cid}/recommendations")
    assert r.status_code == 200
    body = r.json()
    assert body["group_tags"] and body["final_decision"]["recommended"]
    assert len(body["checklist"]) == 10
