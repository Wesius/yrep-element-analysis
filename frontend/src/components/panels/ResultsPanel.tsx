/** Results panel for viewing pipeline execution output */

import { usePipelineStore } from '../../store';
import type { NodeResult } from '../../types';

/** Status badge colors */
const statusColors: Record<string, string> = {
  success: 'bg-green-500/20 text-green-400 border-green-500/30',
  partial: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
  error: 'bg-red-500/20 text-red-400 border-red-500/30',
  skipped: 'bg-slate-500/20 text-slate-400 border-slate-500/30',
};

export function ResultsPanel() {
  const executionResult = usePipelineStore((s) => s.executionResult);
  const isExecuting = usePipelineStore((s) => s.isExecuting);
  const nodes = usePipelineStore((s) => s.nodes);

  if (isExecuting) {
    return (
      <div className="p-4 text-slate-400 text-center">
        <div className="animate-pulse">
          <p className="mb-2 text-lg">Executing Pipeline...</p>
          <p className="text-sm">Processing nodes in order</p>
        </div>
      </div>
    );
  }

  if (!executionResult) {
    return (
      <div className="p-4 text-slate-400 text-center">
        <p className="mb-2">No Results Yet</p>
        <p className="text-sm">Execute your pipeline to see results here</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header with overall status */}
      <div className="p-3 border-b border-slate-700">
        <div className="flex items-center justify-between">
          <h3 className="font-medium text-slate-100">Execution Results</h3>
          <StatusBadge status={executionResult.status} />
        </div>
        {executionResult.error && (
          <p className="text-xs text-red-400 mt-2">{executionResult.error}</p>
        )}
      </div>

      {/* Results list */}
      <div className="flex-1 overflow-y-auto p-3 space-y-3">
        {executionResult.execution_order.map((nodeId, index) => {
          const nodeResult = executionResult.node_results[nodeId];
          const node = nodes.find((n) => n.id === nodeId);
          const nodeName = node?.data.title || nodeId;

          return (
            <NodeResultCard
              key={nodeId}
              nodeId={nodeId}
              nodeName={nodeName}
              result={nodeResult}
              index={index}
            />
          );
        })}
      </div>

      {/* Summary */}
      <div className="p-3 border-t border-slate-700 text-xs text-slate-400">
        <p>
          {Object.values(executionResult.node_results).filter((r) => r.status === 'success').length} succeeded,{' '}
          {Object.values(executionResult.node_results).filter((r) => r.status === 'error').length} failed,{' '}
          {Object.values(executionResult.node_results).filter((r) => r.status === 'skipped').length} skipped
        </p>
      </div>
    </div>
  );
}

/** Status badge component */
function StatusBadge({ status }: { status: string }) {
  const colorClass = statusColors[status] || statusColors.error;
  return (
    <span className={`text-xs px-2 py-1 rounded border ${colorClass}`}>
      {status.charAt(0).toUpperCase() + status.slice(1)}
    </span>
  );
}

/** Individual node result card */
function NodeResultCard({
  nodeId,
  nodeName,
  result,
  index,
}: {
  nodeId: string;
  nodeName: string;
  result: NodeResult;
  index: number;
}) {
  const selectNode = usePipelineStore((s) => s.selectNode);

  return (
    <div
      className="bg-slate-700 rounded-lg p-3 cursor-pointer hover:bg-slate-600 transition-colors"
      onClick={() => selectNode(nodeId)}
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-500 font-mono">#{index + 1}</span>
          <h4 className="text-sm font-medium text-slate-100">{nodeName}</h4>
        </div>
        <StatusBadge status={result.status} />
      </div>

      {result.error && (
        <div className="mt-2 p-2 bg-red-900/20 border border-red-700/30 rounded text-xs text-red-300">
          {result.error}
        </div>
      )}

      {result.output !== undefined && result.output !== null && (
        <ResultOutput output={result.output} />
      )}
    </div>
  );
}

/** Display result output based on type */
function ResultOutput({ output }: { output: unknown }) {
  // Handle different output types
  if (output === null || output === undefined) {
    return null;
  }

  // Check for detection results
  if (isDetectionResult(output)) {
    return (
      <div className="mt-2 space-y-2">
        {output.detections.length > 0 ? (
          <div className="space-y-1">
            <p className="text-xs text-slate-400">Detections:</p>
            {output.detections.slice(0, 5).map((d, i) => (
              <div
                key={i}
                className="flex justify-between items-center text-xs bg-slate-600/50 px-2 py-1 rounded"
              >
                <span className="text-slate-200">{d.species}</span>
                <span className="text-slate-400">
                  {(d.score * 100).toFixed(1)}%
                </span>
              </div>
            ))}
            {output.detections.length > 5 && (
              <p className="text-xs text-slate-500">
                +{output.detections.length - 5} more
              </p>
            )}
          </div>
        ) : (
          <p className="text-xs text-slate-500">No detections found</p>
        )}
      </div>
    );
  }

  // Check for signal data
  if (isSignalData(output)) {
    return (
      <div className="mt-2">
        <p className="text-xs text-slate-400">
          Signal: {output.wavelength.length} data points
        </p>
        <p className="text-xs text-slate-500">
          Range: {Math.min(...output.wavelength).toFixed(1)} -{' '}
          {Math.max(...output.wavelength).toFixed(1)} nm
        </p>
      </div>
    );
  }

  // Check for array results
  if (Array.isArray(output)) {
    return (
      <div className="mt-2">
        <p className="text-xs text-slate-400">{output.length} items</p>
      </div>
    );
  }

  // Check for object with plot data
  if (isPlotData(output)) {
    return (
      <div className="mt-2">
        <p className="text-xs text-slate-400">Plot: {output.title}</p>
        <p className="text-xs text-slate-500">{output.series.length} series</p>
      </div>
    );
  }

  // Default: show as JSON preview
  if (typeof output === 'object') {
    const keys = Object.keys(output as Record<string, unknown>);
    return (
      <div className="mt-2">
        <p className="text-xs text-slate-400">
          Object with {keys.length} properties
        </p>
        <p className="text-xs text-slate-500 truncate">
          {keys.slice(0, 3).join(', ')}
          {keys.length > 3 ? '...' : ''}
        </p>
      </div>
    );
  }

  // Primitive value
  return (
    <div className="mt-2">
      <p className="text-xs text-slate-400 truncate">{String(output)}</p>
    </div>
  );
}

/** Type guard for detection results */
function isDetectionResult(output: unknown): output is { detections: Array<{ species: string; score: number }> } {
  return (
    typeof output === 'object' &&
    output !== null &&
    'detections' in output &&
    Array.isArray((output as { detections: unknown }).detections)
  );
}

/** Type guard for signal data */
function isSignalData(output: unknown): output is { wavelength: number[]; intensity: number[] } {
  return (
    typeof output === 'object' &&
    output !== null &&
    'wavelength' in output &&
    'intensity' in output &&
    Array.isArray((output as { wavelength: unknown }).wavelength)
  );
}

/** Type guard for plot data */
function isPlotData(output: unknown): output is { title: string; series: unknown[] } {
  return (
    typeof output === 'object' &&
    output !== null &&
    'title' in output &&
    'series' in output &&
    Array.isArray((output as { series: unknown }).series)
  );
}
