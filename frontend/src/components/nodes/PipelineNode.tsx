/** Custom pipeline node component for React Flow */

import { memo } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import type { PipelineNodeData, Port } from '../../types';
import { usePipelineStore, useUIStore } from '../../store';

/** Category color mapping */
const categoryColors: Record<string, string> = {
  'I/O': 'bg-emerald-600',
  'Preprocess': 'bg-blue-600',
  'Detection': 'bg-cyan-600',
  'Utilities': 'bg-amber-600',
  'default': 'bg-slate-600',
};

/** Status indicator colors */
const statusColors: Record<string, string> = {
  pending: 'bg-slate-400',
  running: 'bg-yellow-400 animate-pulse',
  completed: 'bg-green-400',
  error: 'bg-red-400',
};

function PipelineNodeComponent({ id, data, selected }: NodeProps) {
  const nodeData = data as PipelineNodeData;
  const selectNode = usePipelineStore((s) => s.selectNode);
  const openHelpModal = useUIStore((s) => s.openHelpModal);
  const setActivePanel = useUIStore((s) => s.setActivePanel);

  const categoryColor = categoryColors[nodeData.category] || categoryColors.default;
  const statusColor = nodeData.status ? statusColors[nodeData.status] : undefined;

  const handleClick = () => {
    selectNode(id);
    setActivePanel('config');
  };

  const handleHelpClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    openHelpModal(nodeData.identifier);
  };

  return (
    <div
      onClick={handleClick}
      className={`
        min-w-[180px] rounded-lg border-2 transition-all cursor-pointer
        ${selected ? 'border-blue-400 shadow-lg shadow-blue-400/30' : 'border-slate-600'}
        bg-slate-800 hover:border-slate-500
      `}
    >
      {/* Header */}
      <div className={`${categoryColor} px-3 py-2 rounded-t-md flex items-center justify-between`}>
        <span className="font-medium text-white text-sm truncate">
          {nodeData.title}
        </span>
        <div className="flex items-center gap-2">
          {statusColor && (
            <span className={`w-2 h-2 rounded-full ${statusColor}`} />
          )}
          <button
            onClick={handleHelpClick}
            className="text-white/70 hover:text-white text-xs"
            title="Help"
          >
            ?
          </button>
        </div>
      </div>

      {/* Body */}
      <div className="p-2">
        <div className="text-xs text-slate-400 mb-2">{nodeData.category}</div>

        {/* Input handles */}
        {nodeData.inputs.map((input: Port, index: number) => (
          <div key={`input-${index}-${input.name}`} className="relative flex items-center mb-1">
            <Handle
              type="target"
              position={Position.Left}
              id={`input-${index}`}
              className="!w-3 !h-3 !bg-blue-400 !border-2 !border-slate-700"
              style={{ top: 'auto', left: -6 }}
            />
            <span className="text-xs text-slate-300 ml-2">{input.name}</span>
            {input.required && (
              <span className="text-red-400 ml-1">*</span>
            )}
          </div>
        ))}

        {/* Output handles */}
        {nodeData.outputs.map((output: Port, index: number) => (
          <div key={`output-${index}-${output.name}`} className="relative flex items-center justify-end mb-1">
            <span className="text-xs text-slate-300 mr-2">{output.name}</span>
            <Handle
              type="source"
              position={Position.Right}
              id={`output-${index}`}
              className="!w-3 !h-3 !bg-green-400 !border-2 !border-slate-700"
              style={{ top: 'auto', right: -6 }}
            />
          </div>
        ))}
      </div>
    </div>
  );
}

export const PipelineNode = memo(PipelineNodeComponent);
