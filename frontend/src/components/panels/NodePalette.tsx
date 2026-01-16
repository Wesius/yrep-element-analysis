/** Node palette panel for dragging nodes onto canvas */

import { useState, useEffect } from 'react';
import { usePipelineStore } from '../../store';
import { nodesAPI } from '../../api/client';
import type { NodeDefinition } from '../../types';

/** Category icon mapping */
const categoryIcons: Record<string, string> = {
  'I/O': '📁',
  'Preprocess': '⚙️',
  'Detection': '🔍',
  'Utilities': '🛠️',
};

export function NodePalette() {
  const [searchQuery, setSearchQuery] = useState('');
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(
    new Set(['I/O', 'Preprocess', 'Detection'])
  );
  const [groupedNodes, setGroupedNodes] = useState<Record<string, NodeDefinition[]>>({});
  const [loading, setLoading] = useState(true);

  const setNodeDefinitions = usePipelineStore((s) => s.setNodeDefinitions);

  // Load nodes on mount
  useEffect(() => {
    const loadNodes = async () => {
      try {
        const result = await nodesAPI.grouped();
        setGroupedNodes(result.groups);
        // Flatten all nodes for the store
        const allNodes = Object.values(result.groups).flat();
        setNodeDefinitions(allNodes);
      } catch (err) {
        console.error('Failed to load nodes:', err);
      } finally {
        setLoading(false);
      }
    };

    loadNodes();
  }, [setNodeDefinitions]);

  const toggleCategory = (category: string) => {
    setExpandedCategories((prev) => {
      const next = new Set(prev);
      if (next.has(category)) {
        next.delete(category);
      } else {
        next.add(category);
      }
      return next;
    });
  };

  const handleDragStart = (e: React.DragEvent, node: NodeDefinition) => {
    e.dataTransfer.setData('application/yrep-node', node.identifier);
    e.dataTransfer.effectAllowed = 'move';
  };

  // Filter nodes by search query (trimmed)
  const trimmedQuery = searchQuery.trim().toLowerCase();
  const filteredGroups = Object.entries(groupedNodes).reduce(
    (acc, [category, nodes]) => {
      if (!trimmedQuery) {
        acc[category] = nodes;
      } else {
        const filtered = nodes.filter(
          (n) =>
            n.title.toLowerCase().includes(trimmedQuery) ||
            n.identifier.toLowerCase().includes(trimmedQuery) ||
            (n.description?.toLowerCase() || '').includes(trimmedQuery)
        );
        if (filtered.length > 0) {
          acc[category] = filtered;
        }
      }
      return acc;
    },
    {} as Record<string, NodeDefinition[]>
  );

  const hasResults = Object.keys(filteredGroups).length > 0;

  if (loading) {
    return (
      <div className="p-4 text-slate-400 text-center">
        Loading nodes...
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Search input */}
      <div className="p-3 border-b border-slate-700">
        <input
          type="text"
          placeholder="Search nodes..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-sm text-slate-100 placeholder-slate-400 focus:outline-none focus:border-blue-500"
        />
      </div>

      {/* Node categories */}
      <div className="flex-1 overflow-y-auto">
        {!hasResults ? (
          <div className="p-4 text-slate-400 text-center">
            <p className="mb-2">No nodes found</p>
            <p className="text-sm">Try a different search term</p>
          </div>
        ) : (
          Object.entries(filteredGroups).map(([category, nodes]) => (
            <div key={category} className="border-b border-slate-700">
              {/* Category header */}
              <button
                onClick={() => toggleCategory(category)}
                className="w-full px-3 py-2 flex items-center justify-between hover:bg-slate-700 transition-colors"
              >
                <span className="flex items-center gap-2 text-sm font-medium text-slate-200">
                  <span>{categoryIcons[category] || '📦'}</span>
                  {category}
                  <span className="text-xs text-slate-400">({nodes.length})</span>
                </span>
                <span className="text-slate-400">
                  {expandedCategories.has(category) ? '▼' : '▶'}
                </span>
              </button>

              {/* Nodes in category */}
              {expandedCategories.has(category) && (
                <div className="pb-2">
                  {nodes.map((node) => (
                    <div
                      key={node.identifier}
                      draggable
                      onDragStart={(e) => handleDragStart(e, node)}
                      className="mx-2 my-1 px-3 py-2 bg-slate-700 rounded cursor-grab hover:bg-slate-600 transition-colors"
                    >
                      <div className="text-sm font-medium text-slate-100">
                        {node.title}
                      </div>
                      <div className="text-xs text-slate-400 truncate">
                        {node.description || 'No description'}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))
        )}
      </div>

      {/* Help text */}
      <div className="p-3 text-xs text-slate-500 border-t border-slate-700">
        Drag nodes onto the canvas to build your pipeline
      </div>
    </div>
  );
}
