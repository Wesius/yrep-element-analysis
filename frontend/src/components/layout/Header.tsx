/** Header component with title and actions */

import { usePipelineStore } from '../../store';
import { pipelinesAPI } from '../../api/client';
import type { PipelineGraphAPI } from '../../types';

/** Extract error message from unknown error */
function getErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
}

/** Safely parse port number from handle string */
function parsePortNumber(handle: string | null | undefined, prefix: string): number {
  if (!handle) return 0;
  const stripped = handle.replace(prefix, '');
  const parsed = parseInt(stripped, 10);
  return Number.isNaN(parsed) ? 0 : parsed;
}

export function Header() {
  const nodes = usePipelineStore((s) => s.nodes);
  const edges = usePipelineStore((s) => s.edges);
  const clearPipeline = usePipelineStore((s) => s.clearPipeline);
  const isExecuting = usePipelineStore((s) => s.isExecuting);
  const setExecuting = usePipelineStore((s) => s.setExecuting);
  const setExecutionResult = usePipelineStore((s) => s.setExecutionResult);
  const updateNodeStatus = usePipelineStore((s) => s.updateNodeStatus);

  const buildGraph = (): PipelineGraphAPI => ({
    version: 1,
    nodes: nodes.map((n) => ({
      id: n.id,
      identifier: n.data.identifier,
      config: n.data.config,
      position: n.position,
    })),
    edges: edges.map((e) => ({
      id: e.id,
      source_node: e.source,
      source_port: parsePortNumber(e.sourceHandle, 'output-'),
      target_node: e.target,
      target_port: parsePortNumber(e.targetHandle, 'input-'),
    })),
  });

  const handleValidate = async () => {
    try {
      const graph = buildGraph();
      const result = await pipelinesAPI.validate(graph);

      if (result.valid) {
        alert('Pipeline is valid!');
      } else {
        alert(`Validation errors:\n${result.errors.join('\n')}`);
      }
    } catch (err) {
      alert(`Validation failed: ${getErrorMessage(err)}`);
    }
  };

  const handleExecute = async () => {
    try {
      setExecuting(true);

      // Reset all node statuses
      nodes.forEach((n) => updateNodeStatus(n.id, 'pending'));

      const graph = buildGraph();
      const result = await pipelinesAPI.execute(graph);
      setExecutionResult(result);

      // Update node statuses based on results
      Object.entries(result.node_results).forEach(([nodeId, nodeResult]) => {
        updateNodeStatus(
          nodeId,
          nodeResult.status === 'success' ? 'completed' : 'error'
        );
      });

      if (result.status === 'error') {
        alert(`Execution error: ${result.error}`);
      }
    } catch (err) {
      alert(`Execution failed: ${getErrorMessage(err)}`);
    } finally {
      setExecuting(false);
    }
  };

  const handleClear = () => {
    if (nodes.length === 0 || confirm('Clear the entire pipeline?')) {
      clearPipeline();
    }
  };

  return (
    <header className="h-14 bg-slate-800 border-b border-slate-700 px-4 flex items-center justify-between">
      {/* Logo and title */}
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-slate-600 rounded-lg flex items-center justify-center text-white font-bold">
          Y
        </div>
        <div>
          <h1 className="text-lg font-semibold text-slate-100">YREP Pipeline Builder</h1>
          <p className="text-xs text-slate-400">Spectral Analysis Workflow Editor</p>
        </div>
      </div>

      {/* Pipeline stats */}
      <div className="flex items-center gap-4 text-sm text-slate-400">
        <span>{nodes.length} nodes</span>
        <span>{edges.length} connections</span>
      </div>

      {/* Actions */}
      <div className="flex items-center gap-2">
        <button
          onClick={handleClear}
          className="px-3 py-2 text-sm text-slate-300 hover:text-white hover:bg-slate-700 rounded transition-colors"
        >
          Clear
        </button>
        <button
          onClick={handleValidate}
          className="px-3 py-2 text-sm text-slate-300 hover:text-white hover:bg-slate-700 rounded transition-colors"
        >
          Validate
        </button>
        <button
          onClick={handleExecute}
          disabled={isExecuting || nodes.length === 0}
          className={`
            px-4 py-2 text-sm font-medium rounded transition-colors
            ${isExecuting || nodes.length === 0
              ? 'bg-slate-600 text-slate-400 cursor-not-allowed'
              : 'bg-blue-600 hover:bg-blue-500 text-white'
            }
          `}
        >
          {isExecuting ? 'Executing...' : 'Execute'}
        </button>
      </div>
    </header>
  );
}
