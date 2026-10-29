"""
=====================================================================
 RAG 챗봇 서버 (Google Cloud Run 배포용)
---------------------------------------------------------------------
 - 프레임워크 : FastAPI
 - LLM        : Gemini (langchain-google-genai 의 ChatGoogleGenerativeAI)
 - 임베딩     : OpenAI text-embedding-3-small (1536차원)
 - 벡터 DB    : Supabase (pgvector) + LangChain SupabaseVectorStore

 [엔드포인트]
   GET  /health  : 서버 상태 확인
   POST /upload  : PDF 업로드 → pypdf 로 텍스트 추출 → 전체 텍스트(+페이지별 텍스트) 반환
   POST /build   : 추출 텍스트 → 청크 분할 → 임베딩 → Supabase 저장 (진행 현황 SSE 스트리밍)
   POST /query   : 질의 확장 → 검색(k=4) → 답변 생성 (SSE 스트리밍)

 [SSE 형식]
   모든 스트리밍 응답은 아래 형식으로 전송된다.
       event: <이벤트 타입>
       data: <JSON 문자열>
       (빈 줄)
=====================================================================
"""

import os
import re
import json
import time
import asyncio
from io import BytesIO
from typing import Any, AsyncGenerator, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from pypdf import PdfReader
from supabase import Client, create_client

# LangChain 관련 import
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore


# =====================================================================
# 1. 환경변수 로드
# =====================================================================
# Cloud Run 에서는 콘솔/CLI 의 "환경 변수" 또는 Secret Manager 로 주입한다.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")   # Gemini 호출용 키
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")   # OpenAI 임베딩용 키
SUPABASE_URL = os.getenv("SUPABASE_URL")       # 예: https://xxxx.supabase.co
SUPABASE_KEY = os.getenv("SUPABASE_KEY")       # service_role 키 권장 (insert/delete 권한 필요)

# 모델명은 요구사항대로 기본값을 두되, 필요하면 환경변수로 바꿀 수 있게 열어둔다.
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
# ---------------------------------------------------------------------
# 임베딩 제공자 선택
#   EMBEDDING_PROVIDER = "google" (기본) → Gemini 임베딩 (GOOGLE_API_KEY 사용)
#   EMBEDDING_PROVIDER = "openai"        → OpenAI text-embedding-3-small (OPENAI_API_KEY 사용)
# 어느 쪽이든 결과는 1536 차원 → supabase.sql 의 vector(1536) 과 일치해야 한다.
# ※ 제공자를 바꾸면 기존 벡터와 호환되지 않으므로 반드시 reset=true 로 다시 빌드할 것
# ---------------------------------------------------------------------
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "google").lower()
EMBEDDING_DIM = 1536
if EMBEDDING_PROVIDER == "openai":
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
else:
    # Gemini 임베딩은 기본 3072 차원 → output_dimensionality=1536 으로 줄여서 저장
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

# Supabase 테이블 / 함수 이름 (supabase.sql 과 동일하게 맞춰야 함)
TABLE_NAME = "documents"
QUERY_NAME = "match_documents"

# 청크 분할 설정
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# 검색 시 가져올 청크 개수
TOP_K = 4

# 임베딩/저장을 몇 개씩 묶어서 처리할지 (진행률 갱신 단위이기도 함)
EMBED_BATCH_SIZE = 50

# ---------------------------------------------------------------------
# 임베딩 속도 제한 (Rate Limit) 대응
#   Gemini 무료 티어는 "텍스트 1개 = 요청 1건"으로 계산해 분당 100건(RPM) 제한이 있다.
#   → 1분에 EMBED_RPM 개 이하로만 보내도록 속도를 조절한다. (여유분 두고 90)
#   유료 티어/OpenAI 처럼 한도가 넉넉하면 EMBED_RPM=0 으로 두면 제한 없이 보낸다.
# ---------------------------------------------------------------------
EMBED_RPM = int(os.getenv("EMBED_RPM", "90" if EMBEDDING_PROVIDER == "google" else "0"))
EMBED_MAX_RETRIES = 6          # 429 발생 시 최대 재시도 횟수
WAIT_TICK = 5                  # 대기 중 SSE 로 남은 시간을 알려주는 간격(초) → 연결 유지 효과도 있음
STORE_BATCH_SIZE = 100


# =====================================================================
# 2. 전역 객체 (지연 초기화)
# =====================================================================
# 환경변수가 하나라도 빠져 있으면 서버 자체가 죽지 않도록,
# 실제로 필요할 때(첫 요청 시) 생성해서 재사용한다.
# → /health 는 키가 없어도 응답 가능하므로 배포 직후 상태 점검이 쉬워진다.
_supabase: Optional[Client] = None
_embeddings: Optional[Any] = None   # OpenAIEmbeddings 또는 GoogleGenerativeAIEmbeddings
_vector_store: Optional[SupabaseVectorStore] = None


def _require_env() -> None:
    """필수 환경변수가 모두 설정돼 있는지 검사. 없으면 500 에러."""
    missing = [
        name
        for name, value in {
            "GOOGLE_API_KEY": GOOGLE_API_KEY,
            # OpenAI 키는 openai 임베딩을 쓸 때만 필수
            **({"OPENAI_API_KEY": OPENAI_API_KEY} if EMBEDDING_PROVIDER == "openai" else {}),
            "SUPABASE_URL": SUPABASE_URL,
            "SUPABASE_KEY": SUPABASE_KEY,
        }.items()
        if not value
    ]
    if missing:
        raise HTTPException(status_code=500, detail=f"환경변수 누락: {', '.join(missing)}")


def get_supabase() -> Client:
    """Supabase 클라이언트 싱글톤."""
    global _supabase
    if _supabase is None:
        _require_env()
        _supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase


def get_embeddings():
    """임베딩 객체 싱글톤 (EMBEDDING_PROVIDER 에 따라 Gemini / OpenAI)."""
    global _embeddings
    if _embeddings is None:
        _require_env()
        if EMBEDDING_PROVIDER == "openai":
            _embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
        else:
            _embeddings = GoogleGenerativeAIEmbeddings(
                model=EMBEDDING_MODEL,
                google_api_key=GOOGLE_API_KEY,
                output_dimensionality=EMBEDDING_DIM,   # DB 컬럼 vector(1536) 에 맞춤
            )
    return _embeddings


def is_rate_limit_error(e: Exception) -> bool:
    """429 / RESOURCE_EXHAUSTED(할당량 초과) 에러인지 판별."""
    msg = str(e)
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg or "rate limit" in msg.lower()


def parse_retry_delay(e: Exception, default: float = 60.0) -> float:
    """
    에러 메시지에서 서버가 알려준 재시도 대기 시간(초)을 꺼낸다.
      예) "Please retry in 57.934480577s."  또는  'retryDelay': '57s'
    못 찾으면 default(60초). 여유로 1초를 더한다.
    """
    msg = str(e)
    m = re.search(r"retry in ([\d.]+)s", msg) or re.search(r"retryDelay['\"]?\s*:\s*['\"]?([\d.]+)s", msg)
    return (float(m.group(1)) if m else default) + 1.0


def get_vector_store() -> SupabaseVectorStore:
    """LangChain SupabaseVectorStore 싱글톤."""
    global _vector_store
    if _vector_store is None:
        _vector_store = SupabaseVectorStore(
            client=get_supabase(),
            embedding=get_embeddings(),
            table_name=TABLE_NAME,
            query_name=QUERY_NAME,
        )
    return _vector_store


def make_llm(streaming: bool = False, callbacks: Optional[list] = None) -> ChatGoogleGenerativeAI:
    """
    Gemini LLM 생성.
    - 질의 확장용(스트리밍 X)과 답변 생성용(스트리밍 O)을 따로 만든다.
    - 답변용 LLM 에만 StreamHandler 콜백을 붙여야
      질의 확장 단계의 토큰이 답변 스트림에 섞이지 않는다.
    """
    _require_env()
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2,          # 문서 기반 답변이므로 창의성보다 정확성 우선
        streaming=streaming,
        callbacks=callbacks,
    )


# =====================================================================
# 3. FastAPI 앱 + CORS
# =====================================================================
app = FastAPI(title="RAG Chatbot Server", version="1.0.0")

# 브라우저에서 직접 fetch 로 호출하므로 CORS 를 전체 허용한다.
# (allow_credentials=True 와 "*" 는 함께 쓸 수 없으므로 False 로 둔다)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)


# =====================================================================
# 4. 공통 유틸: SSE 포맷 & 스트리밍 응답
# =====================================================================
def sse(event: str, data: Any) -> str:
    """
    SSE(Server-Sent Events) 한 건을 문자열로 만든다.
    data 는 항상 JSON 으로 직렬화 → 프론트에서 JSON.parse 로 바로 사용 가능.
    ensure_ascii=False 로 한글이 \\uXXXX 로 깨지지 않게 한다.
    """
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def sse_response(generator: AsyncGenerator[str, None]) -> StreamingResponse:
    """SSE 용 StreamingResponse. 프록시 버퍼링을 막는 헤더를 함께 설정한다."""
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",       # 캐시 금지
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",         # 중간 프록시(nginx 등)의 버퍼링 방지 → 실시간 전송
        },
    )


# =====================================================================
# 5. 요청 바디 스키마 (Pydantic)
# =====================================================================
class PageText(BaseModel):
    """페이지 단위 텍스트 (출처 페이지 번호를 남기기 위해 사용)"""
    page: int
    text: str


class BuildRequest(BaseModel):
    """
    /build 요청 바디.
    - pages 를 보내면 페이지 번호가 메타데이터에 저장된다. (권장: /upload 응답의 pages 그대로 전달)
    - pages 가 없으면 text(전체 텍스트)만으로 빌드한다. (이 경우 페이지 번호는 null)
    """
    text: Optional[str] = None
    pages: Optional[List[PageText]] = None
    filename: Optional[str] = "document.pdf"
    reset: bool = False   # True 면 기존 documents 테이블 데이터를 모두 지우고 새로 빌드


class ChatTurn(BaseModel):
    """대화 기록 한 건. role 은 'user' 또는 'assistant'"""
    role: str
    content: str


class QueryRequest(BaseModel):
    question: str
    chat_history: List[ChatTurn] = Field(default_factory=list)


# =====================================================================
# 6. GET /health
# =====================================================================
@app.get("/health")
async def health():
    """
    서버 상태 확인용.
    키 값 자체는 노출하지 않고 '설정 여부'만 알려준다.
    """
    return {
        "status": "ok",
        "llm_model": LLM_MODEL,
        "embedding_provider": EMBEDDING_PROVIDER,
        "embedding_model": EMBEDDING_MODEL,
        "embed_rpm_limit": EMBED_RPM,
        "env": {
            "GOOGLE_API_KEY": bool(GOOGLE_API_KEY),
            "OPENAI_API_KEY": bool(OPENAI_API_KEY),
            "SUPABASE_URL": bool(SUPABASE_URL),
            "SUPABASE_KEY": bool(SUPABASE_KEY),
        },
    }


# =====================================================================
# 7. POST /upload  (PDF → 텍스트)
# =====================================================================
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """
    multipart/form-data 로 PDF 를 받아 pypdf 로 텍스트를 추출한다.

    프론트 예시:
        const fd = new FormData(); fd.append("file", pdfFile);
        const res = await fetch(`${API}/upload`, { method: "POST", body: fd });

    응답:
        {
          "filename": "...",
          "num_pages": 12,
          "text": "전체 텍스트",
          "pages": [{"page": 1, "text": "..."}, ...]   ← /build 에 그대로 넘기면 페이지 출처가 보존됨
        }
    """
    # 확장자 / MIME 타입으로 간단히 PDF 여부 확인
    if not (file.filename or "").lower().endswith(".pdf") and file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드할 수 있습니다.")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")

    def _extract() -> List[dict]:
        """pypdf 파싱은 CPU 작업이라 별도 스레드에서 실행 (이벤트 루프 블로킹 방지)"""
        reader = PdfReader(BytesIO(raw))
        result = []
        for i, page in enumerate(reader.pages):
            # extract_text() 가 None 을 돌려주는 경우(이미지 페이지 등)를 대비해 "" 처리
            result.append({"page": i + 1, "text": page.extract_text() or ""})
        return result

    try:
        pages = await asyncio.to_thread(_extract)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF 파싱 실패: {e}")

    # 페이지 사이를 빈 줄로 구분해 전체 텍스트를 만든다
    full_text = "\n\n".join(p["text"] for p in pages).strip()
    if not full_text:
        # 스캔본 PDF(이미지)는 텍스트가 없으므로 OCR 이 필요하다는 안내
        raise HTTPException(status_code=422, detail="텍스트를 추출할 수 없습니다. (스캔 이미지 PDF 는 OCR 필요)")

    return {
        "filename": file.filename,
        "num_pages": len(pages),
        "text": full_text,
        "pages": pages,
    }


# =====================================================================
# 8. POST /build  (청크 분할 → 임베딩 → Supabase 저장, 진행 현황 SSE)
# =====================================================================
@app.post("/build")
async def build(req: BuildRequest):
    """
    SSE 이벤트:
        event: progress  data: {"stage": "split"|"embed"|"store", "current": n, "total": N, "message": "..."}
        event: wait      data: {"stage": "embed", "seconds": 남은초, "message": "..."}   ← 속도 제한 대기 중
        event: done      data: {"chunks": N, "message": "..."}
        event: error     data: {"message": "..."}
    """
    _require_env()

    # pages 우선, 없으면 text 하나를 1개 "페이지" 처럼 취급 (페이지 번호는 None)
    if req.pages:
        sources = [(p.page, p.text) for p in req.pages if p.text.strip()]
    elif req.text and req.text.strip():
        sources = [(None, req.text)]
    else:
        raise HTTPException(status_code=400, detail="text 또는 pages 중 하나는 반드시 필요합니다.")

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            # ---------- (0) 기존 데이터 초기화 (옵션) ----------
            if req.reset:
                yield sse("progress", {"stage": "reset", "current": 0, "total": 1, "message": "기존 데이터 삭제 중"})
                # PostgREST 는 조건 없는 delete 를 막으므로 '항상 참'인 조건을 건다
                await asyncio.to_thread(
                    lambda: get_supabase().table(TABLE_NAME).delete()
                    .neq("id", "00000000-0000-0000-0000-000000000000").execute()
                )
                yield sse("progress", {"stage": "reset", "current": 1, "total": 1, "message": "기존 데이터 삭제 완료"})

            # ---------- (1) 청크 분할 ----------
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )
            docs: List[Document] = []
            total_src = len(sources)
            for idx, (page_no, text) in enumerate(sources, start=1):
                # 페이지 단위로 쪼개야 각 청크가 '몇 페이지에서 왔는지' 메타데이터를 가질 수 있다
                chunks = splitter.split_text(text)
                for c in chunks:
                    docs.append(Document(
                        page_content=c,
                        metadata={"source": req.filename, "page": page_no},
                    ))
                yield sse("progress", {
                    "stage": "split", "current": idx, "total": total_src,
                    "message": f"청크 분할 중 ({idx}/{total_src} 페이지, 누적 {len(docs)}개 청크)",
                })
                await asyncio.sleep(0)   # 이벤트 루프에 제어권을 넘겨 즉시 flush 되도록

            total = len(docs)
            if total == 0:
                yield sse("error", {"message": "분할된 청크가 없습니다."})
                return

            # ---------- (2) 임베딩 (속도 제한 + 429 재시도) ----------
            embeddings = get_embeddings()
            vectors: List[List[float]] = []

            # 한 번에 보낼 개수: 분당 한도가 있으면 그 이하로 맞춘다
            batch_size = min(EMBED_BATCH_SIZE, EMBED_RPM) if EMBED_RPM > 0 else EMBED_BATCH_SIZE
            # 슬라이딩 윈도우: 최근 60초 동안 보낸 (시각, 개수) 기록
            sent_log: List[tuple] = []

            async def wait_with_countdown(seconds: float, reason: str):
                """
                seconds 만큼 기다리면서 WAIT_TICK 초마다 wait 이벤트로 남은 시간을 전송.
                → 사용자는 멈춘 게 아니라 기다리는 중임을 알 수 있고,
                  주기적인 전송 덕분에 중간 프록시가 유휴 연결을 끊지 않는다.
                """
                end = time.monotonic() + seconds
                while (left := end - time.monotonic()) > 0:
                    yield sse("wait", {
                        "stage": "embed", "seconds": round(left),
                        "message": f"{reason} — {round(left)}초 후 재개 ({len(vectors)}/{total})",
                    })
                    await asyncio.sleep(min(WAIT_TICK, left))

            for start in range(0, total, batch_size):
                batch = docs[start:start + batch_size]

                # (a) 사전 속도 조절: 이번 배치를 보내면 60초 내 한도를 넘는지 확인
                if EMBED_RPM > 0:
                    now = time.monotonic()
                    sent_log = [(t, n) for t, n in sent_log if now - t < 60]
                    if sum(n for _, n in sent_log) + len(batch) > EMBED_RPM:
                        # 가장 오래된 기록이 60초 창을 벗어날 때까지 대기
                        wait_sec = 60 - (now - sent_log[0][0]) + 0.5
                        async for ev in wait_with_countdown(wait_sec, "분당 임베딩 한도 도달, 대기 중"):
                            yield ev
                        sent_log = []

                # (b) 임베딩 요청 + 429 발생 시 서버가 알려준 시간만큼 기다렸다 재시도
                for attempt in range(1, EMBED_MAX_RETRIES + 1):
                    try:
                        result = await embeddings.aembed_documents([d.page_content for d in batch])
                        break
                    except Exception as e:
                        if not is_rate_limit_error(e) or attempt == EMBED_MAX_RETRIES:
                            raise   # 할당량 문제가 아니거나 재시도 소진 → error 이벤트로
                        delay = parse_retry_delay(e)
                        async for ev in wait_with_countdown(
                            delay, f"429 할당량 초과 (재시도 {attempt}/{EMBED_MAX_RETRIES - 1})"
                        ):
                            yield ev
                        sent_log = []   # 대기 후에는 윈도우가 비워진 것으로 간주

                sent_log.append((time.monotonic(), len(batch)))
                vectors.extend(result)
                yield sse("progress", {
                    "stage": "embed", "current": len(vectors), "total": total,
                    "message": f"임베딩 중 ({len(vectors)}/{total})",
                })

            # ---------- (3) Supabase 저장 ----------
            # 이미 임베딩을 계산했으므로 add_vectors 로 벡터를 직접 넣는다 (중복 임베딩 비용 방지)
            store = get_vector_store()
            saved = 0
            for start in range(0, total, STORE_BATCH_SIZE):
                d_batch = docs[start:start + STORE_BATCH_SIZE]
                v_batch = vectors[start:start + STORE_BATCH_SIZE]
                # supabase-py 는 동기 클라이언트 → 스레드에서 실행
                await asyncio.to_thread(store.add_vectors, v_batch, d_batch)
                saved += len(d_batch)
                yield sse("progress", {
                    "stage": "store", "current": saved, "total": total,
                    "message": f"Supabase 저장 중 ({saved}/{total})",
                })

            yield sse("done", {"chunks": total, "message": f"빌드 완료: 청크 {total}개 저장"})

        except Exception as e:
            # 스트림이 이미 시작된 뒤라 HTTP 상태코드를 바꿀 수 없으므로 error 이벤트로 알린다
            yield sse("error", {"message": f"빌드 실패: {e}"})

    return sse_response(event_stream())


# =====================================================================
# 9. 스트리밍 콜백 핸들러
# =====================================================================
class StreamHandler(BaseCallbackHandler):
    """
    LLM 이 토큰을 하나 생성할 때마다 on_llm_new_token 이 호출된다.
    그 토큰을 asyncio.Queue 에 넣어두면, SSE 제너레이터가 꺼내서 클라이언트로 보낸다.

    [주의] LangChain 은 동기(BaseCallbackHandler) 콜백을 비동기 실행 중에
           별도 스레드(executor)에서 호출할 수 있다.
           asyncio.Queue 는 스레드-세이프하지 않으므로
           loop.call_soon_threadsafe 로 이벤트 루프 스레드에서 put 하도록 한다.
    """

    def __init__(self, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
        self.queue = queue
        self.loop = loop

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        if token:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, token)


# =====================================================================
# 10. 프롬프트 정의
# =====================================================================
# ① 질의 확장 (Query Augmentation) 프롬프트
QUERY_AUG_SYSTEM = (
    "기존의 대화 내용을 활용하여 사용자가 질문한 의도를 파악해서 한문장의 명료한 질문으로 변환하라. "
    "대명사나 이, 저, 그와 같은 표현을 명확한 명사로 표현하라"
)
query_augmentation_prompt = ChatPromptTemplate.from_messages([
    ("system", QUERY_AUG_SYSTEM),
    MessagesPlaceholder("chat_history"),   # 이전 대화가 여기에 삽입됨
    ("human", "{question}"),
])

# ③ 답변 생성 프롬프트
ANSWER_SYSTEM = (
    "너는 문서에 기반해 답변하는 도시 정책 전문가야.\n"
    "반드시 아래 [참고 문서]의 내용만 근거로 한국어로 답변해.\n"
    "[참고 문서]에 질문에 대한 정보가 없으면 추측하지 말고 "
    "\"제공된 문서에서 해당 내용을 찾을 수 없습니다.\"라고 답해.\n"
    "가능하면 근거가 된 페이지 번호를 (p.N) 형식으로 함께 적어줘.\n\n"
    "[참고 문서]\n{context}"
)
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", ANSWER_SYSTEM),
    ("human", "{question}"),
])


def to_lc_messages(history: List[ChatTurn]) -> List[BaseMessage]:
    """프론트에서 받은 [{role, content}] 를 LangChain 메시지 객체로 변환"""
    msgs: List[BaseMessage] = []
    for turn in history:
        if turn.role in ("user", "human"):
            msgs.append(HumanMessage(content=turn.content))
        elif turn.role in ("assistant", "ai", "bot"):
            msgs.append(AIMessage(content=turn.content))
    return msgs


def format_docs(docs: List[Document]) -> str:
    """검색된 청크들을 프롬프트에 넣을 문자열로 합친다 (페이지 번호 포함)"""
    parts = []
    for i, d in enumerate(docs, start=1):
        page = d.metadata.get("page")
        page_label = f"p.{page}" if page is not None else "페이지 정보 없음"
        parts.append(f"[문서 {i} | {page_label}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


# =====================================================================
# 11. POST /query  (질의 확장 → 검색 → 답변 스트리밍)
# =====================================================================
@app.post("/query")
async def query(req: QueryRequest):
    """
    요청 바디:
        {
          "question": "그거 예산은 얼마야?",
          "chat_history": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
          ]
        }

    SSE 이벤트 (순서대로):
        augmented_query : {"query": "확장된 질문"}
        sources         : [{"content": "...", "page": 3, "source": "a.pdf"}, ...]
        token           : {"token": "생성된 토큰"}      ← 여러 번
        done            : {"answer": "전체 답변"}
        error           : {"message": "..."}           ← 오류 시
    """
    _require_env()
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question 이 비어 있습니다.")

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            chat_history = to_lc_messages(req.chat_history)

            # ---------- ① 질의 확장 ----------
            # 스트리밍이 필요 없으므로 콜백 없는 LLM 사용
            query_augmentation_chain = query_augmentation_prompt | make_llm() | StrOutputParser()
            if chat_history:
                augmented_query = (await query_augmentation_chain.ainvoke({
                    "question": req.question,
                    "chat_history": chat_history,
                })).strip()
            else:
                # 첫 질문이면 대화 맥락이 없으므로 LLM 호출 없이 원문을 그대로 사용 (지연·비용 절감)
                augmented_query = req.question.strip()

            yield sse("augmented_query", {"query": augmented_query})

            # ---------- ② Supabase 리트리버 검색 (k=4) ----------
            retriever = get_vector_store().as_retriever(search_kwargs={"k": TOP_K})
            docs: List[Document] = await retriever.ainvoke(augmented_query)

            yield sse("sources", [
                {
                    "content": d.page_content,
                    "page": d.metadata.get("page"),
                    "source": d.metadata.get("source"),
                }
                for d in docs
            ])

            # ---------- ③ + ④ 답변 생성 & 토큰 스트리밍 ----------
            queue: asyncio.Queue = asyncio.Queue()
            handler = StreamHandler(queue, asyncio.get_running_loop())
            answer_llm = make_llm(streaming=True, callbacks=[handler])
            answer_chain = answer_prompt | answer_llm | StrOutputParser()

            context = format_docs(docs) if docs else "(검색된 문서 없음)"

            # LLM 호출을 백그라운드 태스크로 돌리고,
            # 그동안 큐에 쌓이는 토큰을 꺼내서 바로바로 SSE 로 보낸다.
            task = asyncio.create_task(answer_chain.ainvoke({
                "question": augmented_query,
                "context": context,
            }))

            while True:
                # 토큰을 기다리되, 태스크가 끝났는지 주기적으로 확인
                try:
                    token = await asyncio.wait_for(queue.get(), timeout=0.1)
                    yield sse("token", {"token": token})
                except asyncio.TimeoutError:
                    if task.done() and queue.empty():
                        break

            # 태스크에서 예외가 났다면 여기서 다시 발생 → except 로 이동
            answer = task.result()

            # call_soon_threadsafe 로 늦게 들어온 토큰이 남아 있으면 마저 전송
            while not queue.empty():
                yield sse("token", {"token": queue.get_nowait()})

            yield sse("done", {"answer": answer})

        except Exception as e:
            yield sse("error", {"message": f"질의 처리 실패: {e}"})

    return sse_response(event_stream())


# =====================================================================
# 12. 로컬 실행용 엔트리포인트
# =====================================================================
# Cloud Run 은 PORT 환경변수(기본 8080)로 포트를 알려준다.
# Docker 에서는 Dockerfile 의 CMD 로 실행되지만, 로컬에서 `python main.py` 로도 띄울 수 있게 둔다.
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
