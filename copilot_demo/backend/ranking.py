"""item group risk ranking.

=== 외부 시스템 연동 지점 =====================================================
실제 시스템에서는 기존 분석 로직(사내 통계/ranking 엔진)이 change별 item group
risk score를 제공한다. 이 데모에서는 mock_data가 사전계산해 data/ranking.json 에
기록한 값을 로딩만 한다. 실 전환 시 get_ranking_table() 내부를 사내 API 호출로
교체하면 되며, 반환 스키마(RankingRow)는 그대로 유지한다.
============================================================================
"""
from backend import data_store as ds
from backend.models import RankingRow


def get_ranking_table(cid):
    rows = [RankingRow(**r) for r in ds.get_ranking(cid)]
    rows.sort(key=lambda r: r.risk_score, reverse=True)
    return rows
