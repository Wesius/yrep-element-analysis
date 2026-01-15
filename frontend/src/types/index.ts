/** Core types for the YREP pipeline builder */

/** Node configuration field definition */
export interface ConfigField {
  name: string;
  label: string;
  type: 'string' | 'number' | 'boolean' | 'select' | 'file' | 'directory';
  default?: unknown;
  required?: boolean;
  description?: string;
  options?: string[];
}

/** Port definition for node inputs/outputs */
export interface Port {
  name: string;
  type: string;
  required?: boolean;
  description?: string;
}

/** Node definition from the backend */
export interface NodeDefinition {
  identifier: string;
  title: string;
  category: string;
  description: string;
  explanation: string;
  tips: string[];
  inputs: Port[];
  outputs: Port[];
  config_fields: ConfigField[];
}

/** Pipeline node instance data */
export interface PipelineNodeData {
  identifier: string;
  title: string;
  category: string;
  config: Record<string, unknown>;
  inputs: Port[];
  outputs: Port[];
  status?: 'pending' | 'running' | 'completed' | 'error';
  result?: unknown;
  [key: string]: unknown; // Index signature for React Flow compatibility
}

/** Pipeline node for canvas */
export interface PipelineNode {
  id: string;
  type: string;
  position: { x: number; y: number };
  data: PipelineNodeData;
}

/** Pipeline edge */
export interface PipelineEdge {
  id: string;
  source: string;
  target: string;
  sourceHandle?: string;
  targetHandle?: string;
  type?: string;
}

/** Pipeline graph structure for API */
export interface PipelineGraphAPI {
  version: number;
  name?: string;
  nodes: Array<{
    id: string;
    identifier: string;
    config: Record<string, unknown>;
    position: { x: number; y: number };
  }>;
  edges: Array<{
    id: string;
    source_node: string;
    source_port: number;
    target_node: string;
    target_port: number;
  }>;
  meta?: Record<string, unknown>;
}

/** Preset definition */
export interface Preset {
  id: string;
  name: string;
  description: string;
  category: string;
  explanation: string;
  use_cases: string[];
  parameters: PresetParameter[];
}

/** Preset parameter */
export interface PresetParameter {
  name: string;
  label: string;
  type: string;
  default?: unknown;
  required?: boolean;
  description?: string;
  group?: string;
}

/** Signal data */
export interface Signal {
  wavelength: number[];
  intensity: number[];
  meta?: Record<string, unknown>;
}

/** Detection result */
export interface Detection {
  species: string;
  score: number;
  meta?: Record<string, unknown>;
}

/** Detection result with signal */
export interface DetectionResult {
  signal: Signal;
  detections: Detection[];
  meta?: Record<string, unknown>;
}

/** Plot series */
export interface PlotSeries {
  name: string;
  x: number[];
  y: number[];
  type: 'line' | 'scatter' | 'bar' | 'area';
  color?: string;
  opacity?: number;
  lineWidth?: number;
}

/** Plot data for visualization */
export interface PlotData {
  title: string;
  xLabel: string;
  yLabel: string;
  series: PlotSeries[];
  annotations?: PlotAnnotation[];
  showLegend?: boolean;
  showGrid?: boolean;
}

/** Plot annotation */
export interface PlotAnnotation {
  x: number;
  y: number;
  text: string;
  color?: string;
}

/** Visualization response */
export interface VisualizationResponse {
  plot: PlotData;
  summary?: string;
  meta?: Record<string, unknown>;
}

/** Pipeline execution result */
export interface ExecutionResult {
  status: 'success' | 'partial' | 'error';
  node_results: Record<string, NodeResult>;
  execution_order: string[];
  error?: string;
}

/** Node execution result */
export interface NodeResult {
  status: 'success' | 'skipped' | 'error';
  output?: unknown;
  error?: string;
}

/** Bundled reference library */
export interface BundledReference {
  id: string;
  name: string;
  file: string;
  lineCount: number;
  species: string[];
  path: string;
}
