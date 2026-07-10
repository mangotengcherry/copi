from typing import Literal, Optional
from pydantic import BaseModel, Field


# ---------- 데이터 모델 ----------
class ProcessStep(BaseModel):
    level1: str
    level2: str
    level3: str
    path: str


class Change(BaseModel):
    change_id: str
    process_step: ProcessStep
    change_type: str
    change_direction: str
    description_ko: str
    period_start: str
    period_end: str
    lot_count: int
    status: Literal["past", "current"]


class EDSItem(BaseModel):
    item_id: str
    item_group: str
    struct_location: str
    related_bin: str
    baseline_mean: float
    baseline_std: float
    unit: str


class StatResult(BaseModel):
    change_id: str
    item_id: str
    item_group: str
    p_value: float
    q_value: float
    effect_size: float
    shift_direction: Literal["up", "down", "none"]
    is_representative: bool = False
    mean_before: Optional[float] = None
    std_before: Optional[float] = None
    mean_after: Optional[float] = None
    std_after: Optional[float] = None
    n_hist: Optional[int] = None


class RankingFactors(BaseModel):
    stat_significance: float
    effect_magnitude: float
    coverage: float
    historical_recurrence: float


class RankingRow(BaseModel):
    change_id: str
    group: str
    risk_score: float
    risk_level: Literal["High", "Med", "Low"]
    representative_item: str
    factors: RankingFactors


class AffectedGroup(BaseModel):
    group: str
    direction: Literal["up", "down"]


class ReviewCard(BaseModel):
    change_id: str
    final_decision: str
    affected_groups: list[AffectedGroup]
    confounding_review: str
    follow_up_actions: list[str]
    engineer_comment: str
    reviewer: str
    date: str


# ---------- API 응답 모델 ----------
class GroupImpact(BaseModel):
    group: str
    direction: Literal["up", "down"]
    significant_item_count: int


class OverviewResponse(BaseModel):
    change: Change
    total_items_tested: int
    significant_item_count: int
    affected_groups: list[GroupImpact]


class SimilarBreakdown(BaseModel):
    tree: float
    signature: float
    type: float
    text: float


class SimilarCase(BaseModel):
    change_id: str
    description_ko: str
    change_type: str
    process_path: str
    final_decision: str
    total_score: float
    breakdown: SimilarBreakdown


class SimilarResponse(BaseModel):
    candidates: list[SimilarCase]


class ChecklistSection(BaseModel):
    section: str
    items: list[str]


class ChecklistResponse(BaseModel):
    sections: list[ChecklistSection]


# ---------- LLM 출력 모델 ----------
class PriorityGroup(BaseModel):
    group: str
    reason: str


class SimilarCaseInsight(BaseModel):
    change_id: str
    relevance: str


class CopilotSummary(BaseModel):
    overview_summary: str = Field(..., description="한국어 3~4문장")
    priority_groups: list[PriorityGroup]
    confounding_warnings: list[str]
    similar_case_insights: list[SimilarCaseInsight]
    suggested_checks: list[str]


# ---------- 피드백/보고서 ----------
class GroupFeedback(BaseModel):
    group: str
    tags: list[str]


class FeedbackRequest(BaseModel):
    checklist_state: dict[str, bool] = {}
    group_feedback: list[GroupFeedback] = []
    final_decision: Optional[str] = None
    comment: str = ""


class ReportDraftResponse(BaseModel):
    markdown: str
    used_llm: bool


# ---------- 추천(Copilot 추천 → 승인) ----------
class GroupTagRecommendation(BaseModel):
    group: str
    risk_level: str
    recommended_tags: list[str]
    reason: str


class FinalDecisionRecommendation(BaseModel):
    recommended: str
    reason: str


class ChecklistRecommendation(BaseModel):
    section: str
    index: int
    item: str
    auto_checked: bool
    reason: str


class RecommendationResponse(BaseModel):
    group_tags: list[GroupTagRecommendation]
    final_decision: FinalDecisionRecommendation
    checklist: list[ChecklistRecommendation]
