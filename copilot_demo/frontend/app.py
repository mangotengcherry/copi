"""EDS Impact Review Copilot — Streamlit 프론트."""
import os
import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

API_BASE = os.environ.get("COPILOT_API", "http://localhost:8000")

st.set_page_config(page_title="EDS Impact Review Copilot", layout="wide")


def api_get(path):
    return requests.get(f"{API_BASE}{path}", timeout=120).json()


def api_post(path, json=None):
    return requests.post(f"{API_BASE}{path}", json=json, timeout=120).json()


def risk_color(level):
    return {"High": "#e74c3c", "Med": "#f39c12", "Low": "#2ecc71"}.get(level, "#888")


# ---------- 사이드바 ----------
st.sidebar.title("EDS Impact Review Copilot")
try:
    health = api_get("/health")
    st.sidebar.success(f"백엔드 연결됨: {health['status']}")
    changes = api_get("/changes")
except Exception as e:
    st.sidebar.error(f"백엔드 연결 실패: {e}\nuvicorn backend.main:app 를 먼저 실행하세요.")
    st.stop()

changes_sorted = sorted(changes, key=lambda c: (c["status"] != "current", c["change_id"]))
labels = {f"{c['change_id']} · {c['change_direction']} [{c['status']}]": c["change_id"]
          for c in changes_sorted}
sel_label = st.sidebar.selectbox("변경점 선택", list(labels.keys()))
cid = labels[sel_label]

tab1, tab2, tab3 = st.tabs(["📊 리뷰 대시보드", "🤖 Copilot", "📝 리뷰 & 피드백"])

# ---------- Tab1 ----------
with tab1:
    ov = api_get(f"/changes/{cid}/overview")
    ch = ov["change"]
    c1, c2, c3 = st.columns(3)
    c1.metric("변경 유형", ch["change_type"])
    c2.metric("유의 item 수 (q<0.1)", ov["significant_item_count"])
    c3.metric("적용 lot 수", ch["lot_count"])
    st.caption(f"공정 경로: {ch['process_step']['path']} · 기간: {ch['period_start']} ~ {ch['period_end']}")
    st.info(ch["description_ko"])

    st.subheader("Item Group Ranking")
    ranking = api_get(f"/changes/{cid}/ranking")
    df = pd.DataFrame([{
        "group": r["group"], "risk_level": r["risk_level"], "risk_score": r["risk_score"],
        "대표 item": r["representative_item"],
        "stat_sig": r["factors"]["stat_significance"],
        "effect": r["factors"]["effect_magnitude"],
        "coverage": r["factors"]["coverage"],
    } for r in ranking])

    def _style(row):
        return [f"background-color: {risk_color(row['risk_level'])}22"] * len(row)
    st.dataframe(df.style.apply(_style, axis=1), use_container_width=True, hide_index=True)

    st.subheader("대표 Item Before/After 분포")
    groups = [r["group"] for r in ranking]
    sel_group = st.selectbox("Group filter", groups)
    rep_item = next(r["representative_item"] for r in ranking if r["group"] == sel_group)
    st.caption(f"대표 item: {rep_item}")
    hist_data = api_get(f"/changes/{cid}/histogram/{rep_item}")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=hist_data["before"], name="Before", opacity=0.6))
    fig.add_trace(go.Histogram(x=hist_data["after"], name="After", opacity=0.6))
    fig.update_layout(barmode="overlay", height=360, legend_title_text="분포")
    st.plotly_chart(fig, use_container_width=True)

# ---------- Tab2 ----------
with tab2:
    st.subheader("Copilot 요약")
    if st.button("Copilot 요약 생성", type="primary"):
        with st.spinner("요약 생성 중..."):
            summary = api_post(f"/changes/{cid}/copilot_summary")
        st.session_state[f"summary_{cid}"] = summary
    summary = st.session_state.get(f"summary_{cid}")
    if summary:
        if not summary.get("used_llm", False):
            st.warning("LLM 미가동 — rule-based fallback 요약입니다.")
        st.markdown("**개요 요약**")
        st.write(summary["overview_summary"])
        st.markdown("**우선 검토 Group**")
        for p in summary["priority_groups"]:
            st.markdown(f"- **{p['group']}** — {p['reason']}")
        st.markdown("**Confounding 주의**")
        for w in summary["confounding_warnings"]:
            st.markdown(f"- {w}")
        st.markdown("**제안 점검 항목**")
        for c in summary["suggested_checks"]:
            st.markdown(f"- {c}")

    st.subheader("유사 사례 Top-3")
    similar = api_get(f"/changes/{cid}/similar")
    for cand in similar["candidates"]:
        with st.container(border=True):
            st.markdown(f"**{cand['change_id']}** · {cand['process_path']} · "
                        f"판정: {cand['final_decision']} · 유사도 **{cand['total_score']}**")
            st.caption(cand["description_ko"])
            with st.expander("score breakdown"):
                b = cand["breakdown"]
                st.write(pd.DataFrame([{
                    "tree(0.40)": b["tree"], "signature(0.35)": b["signature"],
                    "type(0.15)": b["type"], "text(0.10)": b["text"]}]))
    if summary:
        for ins in summary.get("similar_case_insights", []):
            st.markdown(f"- 🔎 **{ins['change_id']}**: {ins['relevance']}")

# ---------- Tab3 ----------
with tab3:
    checklist = api_get("/checklist")
    ranking = api_get(f"/changes/{cid}/ranking")
    top_groups = ranking[:5]
    tag_options = ["중요", "Noise", "설비의심", "mix의심", "공정가능", "follow-up"]
    final_options = ["공정영향", "설비영향", "Noise", "재검토"]

    # 위젯 키 헬퍼
    def ck_key(section, i):
        return f"{cid}-{section}-{i}"

    def tag_key(group):
        return f"fb-{cid}-{group}"

    final_key = f"final-{cid}"

    # session_state 기본값 초기화 (위젯 생성 전 — 추천 승인 시 여기에 값을 주입)
    for sec in checklist["sections"]:
        for i in range(len(sec["items"])):
            st.session_state.setdefault(ck_key(sec["section"], i), False)
    for r in top_groups:
        st.session_state.setdefault(tag_key(r["group"]), [])
    st.session_state.setdefault(final_key, final_options[0])

    def _apply_tags(gt):
        st.session_state[tag_key(gt["group"])] = [t for t in gt["recommended_tags"] if t in tag_options]

    # ---------- Copilot 추천 패널 (위젯보다 위 — session_state 주입 안전) ----------
    st.subheader("🤖 Copilot 추천 → 승인")
    st.caption("추천값은 ranking·signal·유사사례에서 코드로 산출됩니다(LLM 미개입). 승인만 하면 아래 폼에 채워집니다.")
    if st.button("추천 불러오기"):
        st.session_state[f"recos_{cid}"] = api_get(f"/changes/{cid}/recommendations")
    recos = st.session_state.get(f"recos_{cid}")
    if recos:
        if st.button("✅ 전체 추천 적용", type="primary"):
            for gt in recos["group_tags"]:
                _apply_tags(gt)
            st.session_state[final_key] = recos["final_decision"]["recommended"]
            for ck in recos["checklist"]:
                if ck["auto_checked"]:
                    st.session_state[ck_key(ck["section"], ck["index"])] = True
            st.rerun()

        st.markdown("**Group 태그 추천**")
        for gt in recos["group_tags"]:
            col1, col2 = st.columns([5, 1])
            col1.markdown(
                f"- **{gt['group']}** ({gt['risk_level']}) → `{', '.join(gt['recommended_tags'])}`  \n"
                f"<span style='color:gray;font-size:0.85em'>{gt['reason']}</span>",
                unsafe_allow_html=True)
            if col2.button("적용", key=f"ap-tag-{cid}-{gt['group']}"):
                _apply_tags(gt)
                st.rerun()

        fd = recos["final_decision"]
        col1, col2 = st.columns([5, 1])
        col1.markdown(
            f"**최종 판정 추천** → `{fd['recommended']}`  \n"
            f"<span style='color:gray;font-size:0.85em'>{fd['reason']}</span>",
            unsafe_allow_html=True)
        if col2.button("적용", key=f"ap-final-{cid}"):
            st.session_state[final_key] = fd["recommended"]
            st.rerun()

        st.markdown("**체크리스트 자동체크 추천**")
        for ck in recos["checklist"]:
            mark = "✅ 자동체크" if ck["auto_checked"] else "▫️ 확인필요"
            col1, col2 = st.columns([5, 1])
            col1.markdown(
                f"- [{ck['section']}] {ck['item']} — {mark}  \n"
                f"<span style='color:gray;font-size:0.85em'>{ck['reason']}</span>",
                unsafe_allow_html=True)
            if ck["auto_checked"] and col2.button("적용", key=f"ap-ck-{cid}-{ck['section']}-{ck['index']}"):
                st.session_state[ck_key(ck["section"], ck["index"])] = True
                st.rerun()

    st.divider()

    # ---------- 리뷰 폼 (session_state 값을 읽어 렌더) ----------
    st.subheader("표준 체크리스트")
    checklist_state = {}
    for sec in checklist["sections"]:
        st.markdown(f"**{sec['section']}**")
        for i, item in enumerate(sec["items"]):
            checklist_state[f"{sec['section']}-{i}"] = st.checkbox(item, key=ck_key(sec["section"], i))

    st.subheader("Item Group 피드백")
    group_feedback = []
    for r in top_groups:
        tags = st.multiselect(f"{r['group']} ({r['risk_level']})", tag_options, key=tag_key(r["group"]))
        if tags:
            group_feedback.append({"group": r["group"], "tags": tags})

    final_decision = st.selectbox("최종 판정", final_options, key=final_key)
    comment = st.text_area("자유 코멘트", key=f"comment-{cid}")

    if st.button("피드백 저장", type="primary"):
        res = api_post(f"/changes/{cid}/feedback", json={
            "checklist_state": checklist_state, "group_feedback": group_feedback,
            "final_decision": final_decision, "comment": comment})
        st.success("저장 완료")
        st.json(res.get("structured_comment", {}))

    if st.button("보고서 초안 생성"):
        with st.spinner("보고서 생성 중..."):
            rep = api_post(f"/changes/{cid}/report_draft")
        if not rep.get("used_llm", False):
            st.warning("LLM 미가동 — rule-based 보고서입니다.")
        st.markdown(rep["markdown"])
        st.download_button("보고서 다운로드", rep["markdown"],
                           file_name=f"report_{cid}.md", mime="text/markdown")
