import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import get_settings
from app.core.langsmith import configure_langsmith

app = FastAPI(
    title="LangGraph Helper Agent",
    description="AI-powered assistant for LangGraph and LangChain questions",
    version="1.0.0",
)


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    settings = get_settings()
    configure_langsmith(settings)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
