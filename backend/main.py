from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from contextlib import asynccontextmanager
from backend.rag_pipeline import (
    load_and_prepare_vectorstore,
    setup_rag_chain,
    cleanup_vectorstore,
    JSON_FILE_PATH,
)
import asyncio
import json
import logging
import time
from datetime import datetime
import concurrent.futures
from functools import lru_cache
import hashlib
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global variables
vectorstore = None
rag_chain = None
query_cache: Dict[str, dict] = {}

# Configure asyncio timeout
asyncio.get_event_loop().set_default_executor(
    concurrent.futures.ThreadPoolExecutor(max_workers=4)
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vectorstore, rag_chain
    try:
        logger.info("Initializing RAG backend...")
        vectorstore = load_and_prepare_vectorstore()
        rag_chain = setup_rag_chain(vectorstore)
        logger.info("RAG backend initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize backend: {str(e)}")
        raise
    yield
    logger.info("Shutting down RAG backend...")
    cleanup_vectorstore()


app = FastAPI(
    title="City Info RAG API",
    description="A RAG-based API for answering questions about city information",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"],  # In production, replace with specific hosts
)


# Request/Response Models
class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="The question to ask about the city",
    )


class QueryResponse(BaseModel):
    answer: str = Field(..., description="The answer to the question")
    confidence_score: Optional[float] = Field(
        None, description="Confidence score of the answer"
    )
    processing_time: float = Field(
        ..., description="Time taken to process the query in seconds"
    )
    cached: bool = Field(
        False, description="Whether the response was served from cache"
    )


class SearchRequest(BaseModel):
    query: str = Field(
        ..., min_length=1, max_length=500, description="The search query"
    )
    top_k: int = Field(5, ge=1, le=20, description="Number of results to return")


class SearchResult(BaseModel):
    id: str = Field(..., description="Unique identifier for the result")
    text: str = Field(..., description="The retrieved text content")
    score: float = Field(..., description="Relevance score of the result")


# Cache configuration
CACHE_TTL = 3600  # 1 hour
MAX_CACHE_SIZE = 1000


def get_cache_key(query: str) -> str:
    """Generate a cache key for a query."""
    return hashlib.md5(query.lower().strip().encode()).hexdigest()


@lru_cache(maxsize=MAX_CACHE_SIZE)
def get_cached_response(cache_key: str) -> Optional[dict]:
    """Get a cached response if it exists."""
    return query_cache.get(cache_key)


def cache_response(cache_key: str, response: dict):
    """Cache a response."""
    if len(query_cache) >= MAX_CACHE_SIZE:
        # Remove oldest entry
        query_cache.pop(next(iter(query_cache)))
    query_cache[cache_key] = response


# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.2f}s"
    )
    return response


@app.get("/")
async def root():
    return {
        "message": "Welcome to the City Info RAG API",
        "version": "1.0.0",
        "endpoints": {
            "/query": "POST - Ask questions about the city",
            "/search": "POST - Search through city information",
            "/health": "GET - Check API health",
            "/data": "GET - Get raw city information data",
        },
    }


@app.get("/health")
async def health_check():
    if not rag_chain or not vectorstore:
        raise HTTPException(status_code=503, detail="Service not fully initialized")
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "rag_chain": "initialized",
            "vectorstore": "initialized",
            "cache_size": len(query_cache),
        },
    }


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    if not rag_chain:
        raise HTTPException(status_code=503, detail="RAG chain not initialized")

    # Check cache
    cache_key = get_cache_key(request.query)
    cached_response = get_cached_response(cache_key)
    if cached_response:
        return QueryResponse(**cached_response, cached=True)

    start_time = time.time()
    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, rag_chain.invoke, {"input": request.query}
        )
        answer = response.get("answer", "Sorry, I could not find an answer.")
        confidence_score = response.get("confidence_score", None)
        processing_time = time.time() - start_time

        result = QueryResponse(
            answer=answer,
            confidence_score=confidence_score,
            processing_time=processing_time,
            cached=False,
        )

        # Cache the response
        cache_response(cache_key, result.dict())

        return result
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=List[SearchResult])
async def search_documents(request: SearchRequest):
    if not vectorstore:
        raise HTTPException(status_code=503, detail="Vectorstore not initialized")

    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": request.top_k})
        results = retriever.get_relevant_documents(request.query)

        formatted_results = []
        for idx, doc in enumerate(results):
            formatted_results.append(
                SearchResult(
                    id=f"doc_{idx+1}",
                    text=doc.page_content,
                    score=getattr(doc, "score", 1.0),
                )
            )
        return formatted_results
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data")
async def get_data():
    try:
        with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return JSONResponse(content=data)
    except FileNotFoundError:
        logger.error(f"Knowledge file not found at {JSON_FILE_PATH}")
        raise HTTPException(status_code=404, detail="Knowledge file not found")
    except Exception as e:
        logger.error(f"Error reading knowledge file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
