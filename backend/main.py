"""YREP Spectral Analysis API - FastAPI application."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes import files, nodes, pipelines, presets, visualizations

app = FastAPI(
    title="YREP Spectral Analysis API",
    description="Backend API for YREP spectral analysis pipeline builder",
    version="0.1.0",
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register route modules
app.include_router(files.router, prefix="/api/files", tags=["files"])
app.include_router(nodes.router, prefix="/api/nodes", tags=["nodes"])
app.include_router(pipelines.router, prefix="/api/pipelines", tags=["pipelines"])
app.include_router(presets.router, prefix="/api/presets", tags=["presets"])
app.include_router(visualizations.router, prefix="/api/visualizations", tags=["visualizations"])


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "yrep-api"}


@app.get("/api/health")
async def health():
    """API health check."""
    return {"status": "healthy", "version": app.version}
