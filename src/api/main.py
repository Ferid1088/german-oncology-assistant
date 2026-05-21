from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.observability import configure_logging, configure_observability
from src.api.routes.analytics import router as analytics_router
from src.api.routes.chat import router as chat_router
from src.api.routes.conversations import router as conversations_router

configure_logging()

app = FastAPI(title="Oncology RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition", "X-Trace-Id"],
)

app.include_router(chat_router)
app.include_router(conversations_router)
app.include_router(analytics_router)
configure_observability(app)


@app.get("/health")
async def health():
    return {"status": "ok"}




