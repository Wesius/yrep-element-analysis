/** Main pipeline canvas with React Flow */

import { useCallback, useRef, type DragEvent } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  BackgroundVariant,
  type Node,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import { usePipelineStore } from '../../store';
import { PipelineNode } from '../nodes/PipelineNode';
import type { PipelineNodeData } from '../../types';

/** Node types for React Flow */
const nodeTypes = {
  pipeline: PipelineNode,
};

/** Generate unique ID for new nodes */
let nodeIdCounter = 0;
const generateNodeId = () => `node-${++nodeIdCounter}`;

export function Canvas() {
  const reactFlowWrapper = useRef<HTMLDivElement>(null);

  const nodes = usePipelineStore((s) => s.nodes);
  const edges = usePipelineStore((s) => s.edges);
  const onNodesChange = usePipelineStore((s) => s.onNodesChange);
  const onEdgesChange = usePipelineStore((s) => s.onEdgesChange);
  const onConnect = usePipelineStore((s) => s.onConnect);
  const addNode = usePipelineStore((s) => s.addNode);
  const nodeDefinitions = usePipelineStore((s) => s.nodeDefinitions);

  // Handle drag over for dropping nodes
  const onDragOver = useCallback((event: DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  // Handle dropping new nodes onto canvas
  const onDrop = useCallback(
    (event: DragEvent) => {
      event.preventDefault();

      const nodeIdentifier = event.dataTransfer.getData('application/yrep-node');
      if (!nodeIdentifier) return;

      const nodeDef = nodeDefinitions.find((n) => n.identifier === nodeIdentifier);
      if (!nodeDef) return;

      // Get drop position relative to canvas
      const reactFlowBounds = reactFlowWrapper.current?.getBoundingClientRect();
      if (!reactFlowBounds) return;

      const position = {
        x: event.clientX - reactFlowBounds.left,
        y: event.clientY - reactFlowBounds.top,
      };

      const newNode: Node<PipelineNodeData> = {
        id: generateNodeId(),
        type: 'pipeline',
        position,
        data: {
          identifier: nodeDef.identifier,
          title: nodeDef.title,
          category: nodeDef.category,
          config: Object.fromEntries(
            nodeDef.config_fields.map((f) => [f.name, f.default])
          ),
          inputs: nodeDef.inputs,
          outputs: nodeDef.outputs,
        },
      };

      addNode(newNode);
    },
    [nodeDefinitions, addNode]
  );

  // Minimap node color based on category
  const minimapNodeColor = useCallback((node: Node) => {
    const category = (node.data as PipelineNodeData)?.category;
    switch (category) {
      case 'I/O':
        return '#10b981';
      case 'Preprocess':
        return '#3b82f6';
      case 'Detection':
        return '#8b5cf6';
      case 'Utilities':
        return '#f59e0b';
      default:
        return '#64748b';
    }
  }, []);

  return (
    <div ref={reactFlowWrapper} className="flex-1 h-full">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onDrop={onDrop}
        onDragOver={onDragOver}
        nodeTypes={nodeTypes}
        defaultEdgeOptions={{
          type: 'smoothstep',
          animated: true,
        }}
        fitView
        snapToGrid
        snapGrid={[15, 15]}
        className="bg-slate-900"
      >
        <Background
          variant={BackgroundVariant.Dots}
          gap={20}
          size={1}
          color="#334155"
        />
        <Controls className="!bg-slate-800 !border-slate-600 !shadow-lg" />
        <MiniMap
          nodeColor={minimapNodeColor}
          className="!bg-slate-800 !border-slate-600"
          maskColor="rgba(30, 41, 59, 0.8)"
        />
      </ReactFlow>
    </div>
  );
}
