# 공정 변경점 EDS Impact Review Copilot — 데모 설계

작성일: 2026-07-10
상태: 승인 대기

## 1. 목적 & 범위

반도체 공정 변경점(process change) 발생 시, 해당 변경이 EDS test item의 통계적
분포에 미친 영향을 엔지니어가 리뷰하는 과정을 보조하는 AI Copilot **데모**.
Streamlit(프론트) + FastAPI(백엔드) + Ollama local LLM(gemma3:12b, 사내명 Gemma4 12B).

**데모 범위**: 합성 mock 데이터만으로 전체 플로우가 완결 동작. 실제 사내 DB/모델 불필요.

### 설계 불변식 (반드시 준수)
1. **LLM은 narration/요약만 담당.** 통계값·ranking·유사도는 전부 Python 코드에서
   계산하고, 그 결과를 컨텍스트로 LLM에 전달. LLM이 숫자를 생성하거나 영향 여부를
   판정하는 구조 금지.
2. **LLM 응답은 Pydantic schema로 강제** (Ollama `format=json` + schema). 파싱 실패 시
   1회 재시도 → 그래도 실패면 rule-based 요약 텍스트 fallback.
3. **모든 데이터는 합성 mock.** 결정론적(seed 고정) 생성으로 재현 가능.

### 확정된 설계 결정 (브레인스토밍)
- **데이터 저장 포맷: JSON** (pyarrow 의존성 회피, 사람이 열람 가능, 규모상 성능 무관).
- **LLM 연동: fallback 우선 + 문서화.** 코드는 라이브 경로+fallback 모두 구현하되,
  이번 빌드에서 모델 pull/실행 검증은 하지 않고 README에 절차 문서화. LLM 미가동 시에도
  전체 플로우 동작.
- **현재/과거 구분: 12 과거 + 3 현재.** 12건은 리뷰카드 보유(과거·유사도 후보 풀),
  3건은 리뷰 진행 중(Copilot 데모 주 대상).
- **한국어 텍스트 유사도: 문자 n-gram(2~3) TF-IDF.** 형태소 분석기(KoNLPy=Java 의존)
  대신 의존성 가벼운 char n-gram. 가중치 0.1이라 전체 영향 작음.

## 2. 아키텍처

```
┌────────────┐   HTTP    ┌───────────────────────────────────┐   HTTP   ┌──────────────┐
│ Streamlit  │ ────────► │ FastAPI  (:8000)                  │ ───────► │ Ollama       │
│ (:8501)    │           │  ┌─────────────────────────────┐  │  chat    │ gemma3:12b   │
│  3 tabs    │ ◄──────── │  │ 코드 계산 레이어             │  │ ◄─────── │ (:11434)     │
└────────────┘   JSON    │  │  stats / ranking / similarity│  │          └──────────────┘
                         │  └──────────────┬──────────────┘  │  실패 시        │
                         │       context   │  narration      │  rule-based ◄───┘
                         │                 ▼                  │  fallback
                         │        llm_client → JSON schema    │
                         └───────────────────────────────────┘
```

세 프로세스로 즉시 구동: `ollama serve` · `uvicorn backend.main:app` · `streamlit run frontend/app.py`.

## 3. 프로젝트 구조

```
copilot_demo/
├── backend/
│   ├── __init__.py
│   ├── main.py              # FastAPI 앱 + 엔드포인트
│   ├── models.py            # Pydantic schemas (요청/응답/LLM 출력)
│   ├── mock_data.py         # 합성 데이터 생성 (seed 고정) → data/ 기록
│   ├── data_store.py        # data/ JSON 로딩 + 인메모리 캐시 접근자
│   ├── ranking.py           # item group ranking 로딩 (외부 시스템 연동 지점 주석)
│   ├── similarity.py        # 유사 변경점 검색 (2단계, 가중치 상수 노출)
│   ├── stats_summary.py     # overview 통계요약 코드계산 (histogram 포함)
│   ├── llm_client.py        # Ollama 호출 wrapper + 재시도 + fallback
│   └── prompts.py           # system/user prompt templates
├── frontend/
│   └── app.py               # Streamlit 3-tab UI
├── data/                    # 생성된 mock 데이터 (json) + feedback.jsonl
├── requirements.txt
└── README.md                # 실행 순서 + mermaid 아키텍처 + 전환 지점
```

> 참고: 스펙 원안 대비 `data_store.py`, `stats_summary.py` 2개 모듈을 추가로 분리.
> 각 파일이 단일 책임을 갖도록(로딩 vs 통계 계산 vs LLM) 경계를 명확히 하기 위함.

## 4. Mock 데이터 설계

**결정론**: `mock_data.py`가 `random.seed(42)` + `numpy` Generator(seed=42)로 생성.
실행 시 `data/*.json`을 항상 동일하게 재생성. 서버는 `data_store.py`로 로딩만.

### 4.1 changes — 15건 (`changes.json`)
| 필드 | 설명 |
|---|---|
| `change_id` | `CHG-2024-001` … `CHG-2024-015` |
| `process_step` | `{level1, level2, level3, path}` — level1: FE/BE, level2: module(CVD/Etch/Photo/Implant/CMP/Diffusion/Metal), level3: layer(예 GateOx). path="FE/CVD/GateOx" |
| `change_type` | `recipe_param` \| `hardware` \| `material` \| `route` |
| `change_direction` | 예: "증착온도 +5%", "타겟 소재 교체", "세정 스텝 추가" |
| `description_ko` | 한국어 1~2문장 변경 서술 |
| `period_start` / `period_end` | ISO 날짜 (변경 적용 기간) |
| `lot_count` | 20~60 |
| `status` | `past`(12건) \| `current`(3건) |

### 4.2 eds_items — 200개 (`eds_items.json`)
| 필드 | 설명 |
|---|---|
| `item_id` | `EDS_001` … `EDS_200` |
| `item_group` | 8개 중 하나: WL, Cell_Edge, Peripheral, Contact, Metal, Vth, Leakage, Speed |
| `struct_location` | 구조 위치 문자열(예 "Cell array core") |
| `related_bin` | 예 "BIN5" |
| `baseline_mean` / `baseline_std` / `unit` | 파라미터 baseline 분포(히스토그램·effect 계산용) |

### 4.3 통계 결과 — change × item (`stats.json`)
각 change마다 **signal group 2~3개**를 지정. 신호 구조를 다음과 같이 주입:
- **Signal group의 item**: 방향 group마다 일관(up 또는 down), `effect_size` ∈ [0.4, 1.2](방향 부호 반영),
  `p_value` ∈ [1e-5, 0.02]. `shift_direction` = up/down.
- **비신호 item**: `effect_size` ~ N(0, 0.1), `p_value` ~ Uniform(0.05, 1.0), `shift_direction` = none.
- **`q_value`는 change 단위 Benjamini–Hochberg 보정으로 실제 계산** → 현실적 유의 구조.

| 필드 | 설명 |
|---|---|
| `change_id`, `item_id`, `item_group` | 키 |
| `p_value`, `q_value`, `effect_size`, `shift_direction` | 통계량 |

**Before/After 히스토그램(Tab1)**: raw 샘플을 저장하지 않고, 대표 item별
분포 파라미터(`mean_before`, `std_before`, `mean_after`, `std_after`, `n`)를 `stats.json`에
포함. 요청 시 `stats_summary.py`가 `hash(change_id+item_id)` seed로 샘플을 결정론적 재생성
→ 파일 경량화 + 재현성 확보.

### 4.4 ranking score — change × group (`ranking.json`, 사전계산)
`ranking.py`는 이 파일을 **로딩만** 한다. 파일 상단/함수에
`# === 외부 시스템 연동 지점: 실제로는 기존 분석 로직이 group risk score를 제공 ===` 주석.
통계 신호와 일관되게 생성(signal group → High/Med, 그 외 → Low).

| 필드 | 설명 |
|---|---|
| `change_id`, `group` | 키 (change당 관련 group 행) |
| `risk_score` | 0~1 |
| `risk_level` | High / Med / Low |
| `representative_item` | 대표 item_id |
| `factors` | `{stat_significance, effect_magnitude, coverage, historical_recurrence}` (0~1 근거 딕셔너리) |

### 4.5 과거 리뷰 카드 — 12건 (`review_cards.json`)
`status=past`인 12개 change 각각에 대응.
| 필드 | 설명 |
|---|---|
| `change_id` | 대응 change |
| `final_decision` | 공정영향 / 설비영향 / Noise / 재검토 |
| `affected_groups` | `[{group, direction}]` (주요 영향 group+방향) |
| `confounding_review` | confounding 검토 결과 서술 |
| `follow_up_actions` | `[str]` |
| `engineer_comment` | 엔지니어 코멘트 |
| `reviewer`, `date` | 메타 |

## 5. 유사도 엔진 (`similarity.py`)

후보 풀 = **리뷰카드 보유 change(자기 자신 제외)**. 2단계:

1. **Hard filter**: `process_step.level1` 대분류 일치 **OR** `change_type` 일치.
2. **Weighted score** (가중치를 모듈 상단 상수로 노출):
   - `W_TREE = 0.40` — process_step 트리 근접도: 같은 leaf 1.0 / 같은 module 0.6 / 같은 대분류 0.3 / 그 외 0.0
   - `W_SIGNATURE = 0.35` — 영향 signature 유사도: 유의차(q<0.1) item group set의 **weighted Jaccard**, shift 방향 일치 시 보너스
   - `W_TYPE = 0.15` — change_type 일치(1/0)
   - `W_TEXT = 0.10` — `description_ko`의 문자 n-gram(2~3) TF-IDF cosine

반환: **Top-3** 사례 + 각 축별 score breakdown(설명가능성). 응답에 total_score와
`breakdown{tree, signature, type, text}` 포함.

## 6. FastAPI 엔드포인트 (`main.py`)

| Method | Path | 설명 |
|---|---|---|
| GET | `/changes` | 변경점 목록(요약) |
| GET | `/changes/{id}/overview` | 기본정보 + 통계요약(코드계산: 유의 item 수, 영향 group 등) |
| GET | `/changes/{id}/ranking` | item group ranking table |
| GET | `/changes/{id}/similar` | 유사 사례 Top-3 (score breakdown 포함) |
| POST | `/changes/{id}/copilot_summary` | LLM 요약 — overview+ranking+similar 컨텍스트 조립 → 구조화 JSON |
| GET | `/checklist` | 표준 체크리스트 (물량/통계/공정/최종 4섹션, 정적 정의) |
| POST | `/changes/{id}/feedback` | 피드백 저장 (`data/feedback.jsonl` append) |
| POST | `/changes/{id}/report_draft` | LLM 보고서 초안 (markdown 반환) |

overview/ranking/similar 결과는 서버가 조립해 copilot_summary/report_draft의 컨텍스트로 재사용.

## 7. LLM 통합 (`llm_client.py`, `prompts.py`)

- Endpoint: `http://localhost:11434/api/chat`, model `gemma3:12b`, `temperature 0.2`, `timeout 60s`.
- `format=json` + Pydantic schema 전달. 파싱/검증 실패 → 1회 재시도 → 실패 시 rule-based fallback.
- **copilot_summary 응답 schema**:
  ```
  {
    overview_summary: str            # 한국어 3~4문장
    priority_groups: [{group, reason}]
    confounding_warnings: [str]
    similar_case_insights: [{change_id, relevance: str}]
    suggested_checks: [str]
  }
  ```
- **System prompt 명시**: "제공된 데이터에 없는 수치나 판정을 생성하지 말 것. 공정 영향
  여부를 단정하지 말 것. 모든 주장에 근거 데이터 항목을 언급할 것. 한국어로 답변."
- **Rule-based fallback**: ranking High/Med group, q<0.1 유의 item, Top-1 유사사례 결정을
  템플릿 문장으로 조립. LLM 완전 부재 시에도 동일 schema 형태로 반환.
- **report_draft**: 위 요약 + 리뷰 결과를 markdown 보고서로. 실패 시 rule-based markdown.

## 8. Streamlit UI (`frontend/app.py`)

**사이드바**: 변경점 선택 dropdown(현재 change 우선 노출), 백엔드 상태 표시.

- **Tab1 리뷰 대시보드**: overview 카드, ranking table(risk_level 색상), 대표 item
  before/after 분포 histogram(plotly), item group filter.
- **Tab2 Copilot**: "Copilot 요약 생성" 버튼 → summary JSON 섹션별 렌더링, 유사 사례
  Top-3 카드(score breakdown expander 포함).
- **Tab3 리뷰 & 피드백**: 체크리스트 checkbox(섹션별), item group별 피드백 버튼
  (중요/Noise/설비의심/mix의심/공정가능/follow-up), 최종판정 selectbox, 자유 comment.
  저장 시 LLM이 comment 구조화(실패 시 원문 저장). "보고서 초안 생성" → markdown 표시 + 다운로드.

## 9. 실패/안정성

- 백엔드 미가동: Streamlit이 연결 실패를 감지해 안내 메시지.
- LLM 미가동/타임아웃/파싱실패: rule-based fallback으로 동일 schema 반환 → 전체 플로우 유지.
- 데이터 파일 부재: 서버 기동 시 `mock_data.py`를 자동 실행해 생성(또는 README에 생성 명령 안내).

## 10. 테스트 (데모 수준)

- `mock_data` 생성 결정론성(seed 고정 시 동일 산출) 스모크 테스트.
- similarity: hard filter/가중합/Top-3 반환 형태 검증(합성 케이스).
- llm_client: Ollama mock으로 파싱 실패→재시도→fallback 경로 검증.
- 엔드포인트: FastAPI TestClient로 각 GET/POST 200 + schema 검증.

## 11. README 필수 내용

- 실행 순서: `python -m backend.mock_data` → `ollama pull gemma3:12b` → `ollama serve` →
  `uvicorn backend.main:app` → `streamlit run frontend/app.py`.
- mermaid 아키텍처 다이어그램.
- **실제 시스템 전환 시 교체 지점**:
  - mock 데이터(`mock_data.py`/`data/`) → 실 DB
  - ranking 사전계산 로딩(`ranking.py` 외부 연동 지점) → 실 분석 로직 연동
  - TF-IDF 텍스트 유사도 → embedding 모델
  - `feedback.jsonl` → feedback DB
  - Streamlit → Vue

## 12. 환경 참고

- Python 3.14.5 감지됨. venv 사용 + requirements 버전 핀. 특정 패키지 3.14 wheel 부재 시
  호환 Python(예 3.12)으로 폴백하는 절차를 README에 병기.
