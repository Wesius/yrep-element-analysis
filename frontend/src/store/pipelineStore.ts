/** Zustand store for pipeline state management */

import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import {
  addEdge,
  applyNodeChanges,
  applyEdgeChanges,
  type Connection,
  type NodeChange,
  type EdgeChange,
  type Node,
  type Edge,
} from '@xyflow/react';
import type { PipelineNodeData, NodeDefinition, ExecutionResult } from '../types';

interface PipelineState {
  // Pipeline graph
  nodes: Node<PipelineNodeData>[];
  edges: Edge[];

  // Node registry
  nodeDefinitions: NodeDefinition[];

  // UI state
  selectedNodeId: string | null;
  isPanelOpen: boolean;

  // Execution state
  isExecuting: boolean;
  executionResult: ExecutionResult | null;

  // Actions
  setNodes: (nodes: Node<PipelineNodeData>[]) => void;
  setEdges: (edges: Edge[]) => void;
  onNodesChange: (changes: NodeChange[]) => void;
  onEdgesChange: (changes: EdgeChange[]) => void;
  onConnect: (connection: Connection) => void;

  addNode: (node: Node<PipelineNodeData>) => void;
  updateNodeConfig: (nodeId: string, config: Record<string, unknown>) => void;
  deleteNode: (nodeId: string) => void;

  setNodeDefinitions: (definitions: NodeDefinition[]) => void;

  selectNode: (nodeId: string | null) => void;
  togglePanel: () => void;

  setExecuting: (isExecuting: boolean) => void;
  setExecutionResult: (result: ExecutionResult | null) => void;
  updateNodeStatus: (nodeId: string, status: PipelineNodeData['status']) => void;

  clearPipeline: () => void;
  loadPipeline: (nodes: Node<PipelineNodeData>[], edges: Edge[]) => void;
}

export const usePipelineStore = create<PipelineState>()(
  devtools(
    (set, get) => ({
      // Initial state
      nodes: [],
      edges: [],
      nodeDefinitions: [],
      selectedNodeId: null,
      isPanelOpen: true,
      isExecuting: false,
      executionResult: null,

      // Node and edge setters
      setNodes: (nodes) => set({ nodes }),
      setEdges: (edges) => set({ edges }),

      // React Flow change handlers
      onNodesChange: (changes) => {
        set({
          nodes: applyNodeChanges(changes, get().nodes) as Node<PipelineNodeData>[],
        });
      },

      onEdgesChange: (changes) => {
        set({
          edges: applyEdgeChanges(changes, get().edges),
        });
      },

      onConnect: (connection) => {
        set({
          edges: addEdge(
            {
              ...connection,
              id: `e-${connection.source}-${connection.target}`,
              type: 'smoothstep',
            },
            get().edges
          ),
        });
      },

      // Node operations
      addNode: (node) => {
        set({ nodes: [...get().nodes, node] });
      },

      updateNodeConfig: (nodeId, config) => {
        set({
          nodes: get().nodes.map((node) =>
            node.id === nodeId
              ? { ...node, data: { ...node.data, config } }
              : node
          ),
        });
      },

      deleteNode: (nodeId) => {
        set({
          nodes: get().nodes.filter((n) => n.id !== nodeId),
          edges: get().edges.filter(
            (e) => e.source !== nodeId && e.target !== nodeId
          ),
          selectedNodeId: get().selectedNodeId === nodeId ? null : get().selectedNodeId,
        });
      },

      // Registry
      setNodeDefinitions: (definitions) => set({ nodeDefinitions: definitions }),

      // UI state
      selectNode: (nodeId) => set({ selectedNodeId: nodeId }),
      togglePanel: () => set({ isPanelOpen: !get().isPanelOpen }),

      // Execution
      setExecuting: (isExecuting) => set({ isExecuting }),
      setExecutionResult: (result) => set({ executionResult: result }),

      updateNodeStatus: (nodeId, status) => {
        set({
          nodes: get().nodes.map((node) =>
            node.id === nodeId
              ? { ...node, data: { ...node.data, status } }
              : node
          ),
        });
      },

      // Pipeline operations
      clearPipeline: () => {
        set({
          nodes: [],
          edges: [],
          selectedNodeId: null,
          executionResult: null,
        });
      },

      loadPipeline: (nodes, edges) => {
        set({
          nodes,
          edges,
          selectedNodeId: null,
          executionResult: null,
        });
      },
    }),
    { name: 'pipeline-store' }
  )
);
