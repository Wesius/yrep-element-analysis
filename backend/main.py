"""YREP Spectral Analysis API - FastAPI application."""

import os
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes import export, files, nodes, pipelines, presets, references, visualizations


def _get_cors_origins() -> List[str]:
    """Get allowed CORS origins from environment or defaults.

    Configure via YREP_CORS_ORIGINS environment variable (comma-separated).
    """
    env_origins = os.environ.get("YREP_CORS_ORIGINS", "")
    if env_origins:
        return [origin.strip() for origin in env_origins.split(",") if origin.strip()]

    # Default: allow localhost development servers
    return [
        "http://localhost:5173",  # Vite default
        "http://localhost:3000",  # Common React default
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ]


app = FastAPI(
    title="YREP Spectral Analysis API",
    description="Backend API for YREP spectral analysis pipeline builder",
    version="0.1.0",
)

# Configure CORS for frontend access with explicit origins
# Using wildcard with credentials is insecure and rejected by browsers
app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_cors_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
)

# Register route modules
app.include_router(export.router, prefix="/api/export", tags=["export"])
app.include_router(files.router, prefix="/api/files", tags=["files"])
app.include_router(nodes.router, prefix="/api/nodes", tags=["nodes"])
app.include_router(pipelines.router, prefix="/api/pipelines", tags=["pipelines"])
app.include_router(presets.router, prefix="/api/presets", tags=["presets"])
app.include_router(references.router, prefix="/api/references", tags=["references"])
app.include_router(visualizations.router, prefix="/api/visualizations", tags=["visualizations"])


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "yrep-api"}


@app.get("/api/health")
async def health():
    """API health check."""
    return {"status": "healthy", "version": app.version}
