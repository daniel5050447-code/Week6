-- =========================================================
-- Supabase 초기 설정 (SQL Editor 에서 한 번 실행)
-- LangChain SupabaseVectorStore 호환 스키마
-- =========================================================

-- 1) pgvector 확장 활성화 (벡터 타입/연산자 제공)
create extension if not exists vector;

-- 2) 문서 청크 저장 테이블
--    id        : LangChain 이 uuid 문자열을 생성해서 넣는다
--    content   : 청크 원문
--    metadata  : {"source": "파일명.pdf", "page": 3} 등
--    embedding : text-embedding-3-small → 1536 차원
create table if not exists documents (
  id uuid primary key default gen_random_uuid(),
  content text,
  metadata jsonb,
  embedding vector(1536)
);

-- 3) 유사도 검색 함수
--    LangChain 은 rpc('match_documents', {query_embedding, filter}) 를 호출하고
--    개수 제한(k)은 PostgREST 의 limit 으로 붙인다. → 함수에 match_count 인자가 필요 없음
--    <=> 는 코사인 거리, 1 - 거리 = 코사인 유사도
create or replace function match_documents (
  query_embedding vector(1536),
  filter jsonb default '{}'
) returns table (
  id uuid,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
#variable_conflict use_column
begin
  return query
  select
    id,
    content,
    metadata,
    1 - (documents.embedding <=> query_embedding) as similarity
  from documents
  where metadata @> filter          -- 메타데이터 필터 (기본 '{}' 이면 전체)
  order by documents.embedding <=> query_embedding
  ;
end;
$$;

-- 4) (선택) 데이터가 많아지면 검색 속도를 위해 HNSW 인덱스 생성
create index if not exists documents_embedding_idx
  on documents using hnsw (embedding vector_cosine_ops);
