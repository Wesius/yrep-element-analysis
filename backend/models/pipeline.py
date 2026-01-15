"""Pipeline graph models for execution and persistence."""

from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field


class PipelineEdge(BaseModel):
    """Connection between two nodes in a pipeline.

    Edges connect output ports of one node to input ports of another,
    defining the data flow through the pipeline.
    """
    id: str = Field(description="Unique edge ID")
    source_node: str = Field(description="Source node instance ID")
    source_port: int = Field(description="Source output port index")
    target_node: str = Field(description="Target node instance ID")
    target_port: int = Field(description="Target input port index")


class PipelineNode(BaseModel):
    """Node instance within a pipeline graph."""
    id: str = Field(description="Unique instance ID")
    identifier: str = Field(description="Node type identifier")
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Node configuration"
    )
    position: Dict[str, float] = Field(
        default_factory=lambda: {"x": 0, "y": 0},
        description="Canvas position"
    )


class PipelineGraph(BaseModel):
    """Complete pipeline graph definition.

    Represents a full analysis workflow that can be saved, loaded,
    and executed. Contains nodes and their connections.
    """
    version: int = Field(
        default=1,
        description="Graph format version"
    )
    name: Optional[str] = Field(
        default=None,
        description="Pipeline name"
    )
    description: Optional[str] = Field(
        default=None,
        description="Pipeline description"
    )
    nodes: List[PipelineNode] = Field(
        default_factory=list,
        description="Node instances in the graph"
    )
    edges: List[PipelineEdge] = Field(
        default_factory=list,
        description="Connections between nodes"
    )
    meta: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "version": 1,
                "name": "Basic Detection Pipeline",
                "nodes": [
                    {"id": "1", "identifier": "load_signal", "config": {"path": "sample.txt"}, "position": {"x": 100, "y": 100}},
                    {"id": "2", "identifier": "trim", "config": {"min_nm": 300, "max_nm": 600}, "position": {"x": 300, "y": 100}}
                ],
                "edges": [
                    {"id": "e1", "source_node": "1", "source_port": 0, "target_node": "2", "target_port": 0}
                ]
            }
        }


class PipelineExecutionRequest(BaseModel):
    """Request to execute a pipeline."""
    graph: PipelineGraph = Field(description="Pipeline graph to execute")
    workspace_root: Optional[str] = Field(
        default=None,
        description="Root directory for resolving relative paths"
    )


class NodeExecutionResult(BaseModel):
    """Result from executing a single node."""
    node_id: str = Field(description="Node instance ID")
    status: Literal["success", "error", "skipped"] = Field(
        description="Execution status"
    )
    output_type: Optional[str] = Field(
        default=None,
        description="Type of output produced"
    )
    output_summary: Optional[str] = Field(
        default=None,
        description="Human-readable output summary"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if status is 'error'"
    )
    duration_ms: Optional[float] = Field(
        default=None,
        description="Execution time in milliseconds"
    )


class PipelineExecutionResult(BaseModel):
    """Result from executing a complete pipeline."""
    status: Literal["success", "partial", "error"] = Field(
        description="Overall execution status"
    )
    node_results: List[NodeExecutionResult] = Field(
        default_factory=list,
        description="Per-node execution results"
    )
    terminal_outputs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Outputs from terminal (sink) nodes"
    )
    execution_order: List[str] = Field(
        default_factory=list,
        description="Order in which nodes were executed"
    )
    total_duration_ms: Optional[float] = Field(
        default=None,
        description="Total execution time in milliseconds"
    )
    error: Optional[str] = Field(
        default=None,
        description="Overall error message if status is 'error'"
    )
