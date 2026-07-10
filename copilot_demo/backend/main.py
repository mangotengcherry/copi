"""FastAPI 앱. 코드 계산 결과를 조립해 LLM 컨텍스트로 전달."""
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend import data_store as ds, ranking as ranking_mod, similarity, llm_client
from backend.stats_summary import build_overview, histogram_samples
from backend.recommend import build_recommendations
from backend.constants import CHECKLIST, DATA_DIR
from backend.models import (
    Change, OverviewResponse, SimilarResponse, ChecklistResponse, ChecklistSection,
    FeedbackRequest, ReportDraftResponse, RecommendationResponse,
)

FEEDBACK_FILE = DATA_DIR / "feedback.jsonl"


@asynccontextmanager
async def lifespan(app: FastAPI):
    ds.ensure_data()
    yield


app = FastAPI(title="EDS Impact Review Copilot", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _require_change(cid):
    c = ds.get_change(cid)
    if c is None:
        raise HTTPException(status_code=404, detail=f"unknown change_id: {cid}")
    return c


def _build_context(cid):
    ov = build_overview(cid)
    table = ranking_mod.get_ranking_table(cid)
    similar = similarity.find_similar(cid)
    return {
        "overview": {
            "change_id": cid,
            "change_direction": ov.change.change_direction,
            "process_path": ov.change.process_step.path,
            "significant_item_count": ov.significant_item_count,
            "affected_groups": [g.model_dump() for g in ov.affected_groups],
        },
        "ranking": [r.model_dump() for r in table],
        "similar": [c.model_dump() for c in similar.candidates],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/changes", response_model=list[Change])
def list_changes():
    return [Change(**c) for c in ds.get_changes()]


@app.get("/changes/{cid}/overview", response_model=OverviewResponse)
def overview(cid: str):
    _require_change(cid)
    return build_overview(cid)


@app.get("/changes/{cid}/ranking")
def ranking(cid: str):
    _require_change(cid)
    return [r.model_dump() for r in ranking_mod.get_ranking_table(cid)]


@app.get("/changes/{cid}/similar", response_model=SimilarResponse)
def similar(cid: str):
    _require_change(cid)
    return similarity.find_similar(cid)


@app.get("/changes/{cid}/histogram/{item_id}")
def histogram(cid: str, item_id: str):
    _require_change(cid)
    return histogram_samples(cid, item_id)


@app.post("/changes/{cid}/copilot_summary")
def copilot_summary(cid: str):
    _require_change(cid)
    context = _build_context(cid)
    summary, used_llm = llm_client.generate_copilot_summary(context)
    body = summary.model_dump()
    body["used_llm"] = used_llm
    return body


@app.get("/checklist", response_model=ChecklistResponse)
def checklist():
    return ChecklistResponse(sections=[ChecklistSection(**s) for s in CHECKLIST])


@app.get("/changes/{cid}/recommendations", response_model=RecommendationResponse)
def recommendations(cid: str):
    _require_change(cid)
    return build_recommendations(cid)


@app.post("/changes/{cid}/feedback")
def feedback(cid: str, req: FeedbackRequest):
    _require_change(cid)
    structured = llm_client.structure_comment(req.comment)
    record = {"change_id": cid, **req.model_dump(), "structured_comment": structured}
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with FEEDBACK_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"saved": True, "structured_comment": structured}


@app.post("/changes/{cid}/report_draft", response_model=ReportDraftResponse)
def report_draft(cid: str):
    _require_change(cid)
    context = _build_context(cid)
    md, used_llm = llm_client.generate_report(context)
    return ReportDraftResponse(markdown=md, used_llm=used_llm)
