"""
Main FastAPI application entry point.
All routers are registered here.
Middleware (CORS, logging) is configured here.
"""

import logging  #track events, errors, and messages while your program runs
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings


# ── Configure logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,   #gives all the erros and login info 
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",  #date format 
)
logger = logging.getLogger(__name__) # this extract the name of the user

settings = get_settings()


# ── Lifespan: startup / shutdown logic ────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):   #fastapi lifespan function which executes before the app start and after the end 
    """
    Code before `yield` runs on startup.
    Code after `yield` runs on shutdown.
    """
    logger.info("=" * 55)
    logger.info(" AI Research Assistant API starting up...")
    settings.ensure_dirs()
    logger.info(f"📁 Docs directory    : {settings.docs_dir}")
    logger.info(f"📁 Vector DB directory: {settings.vector_db_dir}")
    logger.info(f"🤖 LLM Model         : {settings.groq_model}")
    logger.info(f"🔢 Embedding Model   : {settings.embedding_model}")
    logger.info(f"  CORS origins  : {settings.allowed_origins_list}")
    logger.info("=" * 55)
    yield
    logger.info("🛑 AI Research Assistant API shutting down...")


# ── Create FastAPI app ─────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Research Assistant API",
    description=(
        "A production-grade RAG system with memory, "
        "summarization, and document classification."
    ),
    version="1.0.0",
    docs_url="/docs", #just like the swaager, it keeps record of the api endpoints, port:/docs
    redoc_url="/redoc",  #only for reading the well structure and clean endpoints
    lifespan=lifespan,
)


# ── CORS Middleware ────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time"],

)


# ── Request timing middleware ──────────────────────────────────────────────────
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start_time
    response.headers["X-Process-Time"] = f"{duration:.4f}s"
    logger.info(
        f"{request.method} {request.url.path} "
        f"→ {response.status_code} ({duration:.4f}s)"
    )
    return response


# ── Global exception handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred. Please try again.",
        },
    )


# ── Register routers (will be added in later steps) ───────────────────────────
from app.routes import upload          
app.include_router(                    # ← ADD THIS
    upload.router,                     # ← ADD THIS
    prefix="/api/v1",                  # ← ADD THIS
    tags=["Upload"],                   # ← ADD THIS
)                                      # ← ADD THIS
from app.routes import query, analysis 
# app.include_router(upload.router, prefix="/api/v1", tags=["Upload"])
# app.include_router(query.router,  prefix="/api/v1", tags=["Query"])

app.include_router(                           # ← ADD THIS BLOCK
    query.router,
    prefix="/api/v1",
    tags=["Query"],
)

app.include_router(
    analysis.router,
    prefix="/api/v1",
    tags=["Analysis"],
)

# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "AI Research Assistant API",
        "status": "running",
        "version": "1.0.0",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "model": settings.groq_model,
        "embedding_model": settings.embedding_model,
        "vector_db": settings.vector_db_type,
    }