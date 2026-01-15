"""Node registry API routes."""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query

from backend.models.nodes import NodeDefinition
from backend.nodes import (
    get_node_definition,
    get_all_definitions,
    get_definitions_by_category,
    get_category_order,
)

router = APIRouter()


@router.get("/", response_model=List[NodeDefinition])
async def list_nodes(
    category: Optional[str] = Query(None, description="Filter by category"),
):
    """List all available node types.

    Returns node definitions with full metadata including educational content.
    Optionally filter by category.
    """
    if category:
        by_category = get_definitions_by_category()
        if category not in by_category:
            raise HTTPException(
                status_code=404,
                detail=f"Category not found: {category}. Available: {list(by_category.keys())}"
            )
        return by_category[category]

    return get_all_definitions()


@router.get("/categories")
async def list_categories():
    """List node categories in display order.

    Returns categories with their node counts.
    """
    by_category = get_definitions_by_category()
    order = get_category_order()

    return {
        "categories": [
            {
                "name": cat,
                "count": len(by_category.get(cat, [])),
            }
            for cat in order
        ],
        "order": order,
    }


@router.get("/grouped")
async def get_grouped_nodes():
    """Get nodes grouped by category.

    Returns all nodes organized by category in display order.
    Ideal for rendering a node palette/library.
    """
    by_category = get_definitions_by_category()
    order = get_category_order()

    return {
        "categories": order,
        "groups": {
            cat: [node.model_dump() for node in by_category.get(cat, [])]
            for cat in order
        },
    }


@router.get("/{identifier}", response_model=NodeDefinition)
async def get_node(identifier: str):
    """Get a specific node definition by identifier.

    Returns full node definition including educational content.
    """
    node = get_node_definition(identifier)
    if not node:
        available = [n.identifier for n in get_all_definitions()]
        raise HTTPException(
            status_code=404,
            detail=f"Node not found: {identifier}. Available: {available}"
        )
    return node


@router.get("/{identifier}/help")
async def get_node_help(identifier: str):
    """Get educational content for a node.

    Returns explanation, tips, and related nodes for learning.
    """
    node = get_node_definition(identifier)
    if not node:
        raise HTTPException(status_code=404, detail=f"Node not found: {identifier}")

    return {
        "identifier": node.identifier,
        "title": node.title,
        "description": node.description,
        "explanation": node.explanation,
        "tips": node.tips,
        "related_nodes": node.related_nodes,
        "category": node.category,
    }


@router.get("/search/{query}")
async def search_nodes(query: str):
    """Search nodes by title, description, or identifier.

    Returns matching nodes ranked by relevance.
    """
    query_lower = query.lower()
    results = []

    for node in get_all_definitions():
        score = 0

        # Exact identifier match
        if node.identifier == query_lower:
            score += 100

        # Title contains query
        if query_lower in node.title.lower():
            score += 50

        # Description contains query
        if query_lower in node.description.lower():
            score += 20

        # Explanation contains query
        if query_lower in node.explanation.lower():
            score += 10

        # Category matches
        if query_lower in node.category.lower():
            score += 5

        if score > 0:
            results.append({
                "node": node,
                "score": score,
            })

    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)

    return {
        "query": query,
        "results": [r["node"] for r in results],
        "count": len(results),
    }
