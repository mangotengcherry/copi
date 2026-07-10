#!/bin/bash
# 프리뷰/데모용 통합 실행: 빈 포트에 백엔드를 띄우고 Streamlit(8557)을 연결.
# 실제 배포용이 아니라 로컬 확인용 헬퍼. 정식 실행은 README의 3-프로세스 방식을 따른다.
set -e
cd "$(dirname "$0")/.."   # -> copilot_demo
source .venv/bin/activate
BPORT=$(python -c "import socket;s=socket.socket();s.bind(('127.0.0.1',0));print(s.getsockname()[1]);s.close()")
python -m uvicorn backend.main:app --port "$BPORT" &
UV=$!
trap "kill $UV 2>/dev/null" EXIT INT TERM
export COPILOT_API="http://localhost:$BPORT"
for i in $(seq 1 40); do curl -s "localhost:$BPORT/health" | grep -q ok && break; sleep 0.3; done
streamlit run frontend/app.py --server.port 8557 --server.headless true --server.runOnSave false
