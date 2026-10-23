# =========================================================
# Cloud Run 배포용 Dockerfile
# =========================================================
FROM python:3.11-slim

# 파이썬 기본 설정
#  - PYTHONDONTWRITEBYTECODE : .pyc 파일 생성 안 함
#  - PYTHONUNBUFFERED        : 로그가 버퍼링 없이 바로 출력 → Cloud Logging 에서 즉시 확인 가능
#  - PORT                    : Cloud Run 이 주입하지만, 로컬 docker run 용 기본값 8080
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

# 의존성 먼저 설치 → 코드만 바뀔 때는 이 레이어가 캐시되어 빌드가 빨라진다
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# 애플리케이션 코드 복사
COPY main.py .

# 보안을 위해 root 가 아닌 일반 사용자로 실행
RUN useradd --create-home appuser
USER appuser

EXPOSE 8080

# shell 형식으로 실행해야 ${PORT} 환경변수가 치환된다.
# exec 를 붙여 uvicorn 이 PID 1 이 되도록 → Cloud Run 의 종료 신호(SIGTERM)를 정상 수신
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT} --timeout-keep-alive 75
