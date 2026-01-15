/** API client for backend communication */

import type {
  NodeDefinition,
  Preset,
  PipelineGraphAPI,
  ExecutionResult,
  BundledReference,
  VisualizationResponse,
} from '../types';

const API_BASE = '/api';

/** Fetch wrapper with error handling */
async function fetchAPI<T>(
  endpoint: string,
  options?: RequestInit
): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
    ...options,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

/** Nodes API */
export const nodesAPI = {
  list: () => fetchAPI<NodeDefinition[]>('/nodes/'),

  listByCategory: (category: string) =>
    fetchAPI<NodeDefinition[]>(`/nodes/?category=${encodeURIComponent(category)}`),

  get: (identifier: string) =>
    fetchAPI<NodeDefinition>(`/nodes/${identifier}`),

  getHelp: (identifier: string) =>
    fetchAPI<NodeDefinition>(`/nodes/${identifier}/help`),

  search: (query: string) =>
    fetchAPI<{ query: string; count: number; results: NodeDefinition[] }>(
      `/nodes/search/${encodeURIComponent(query)}`
    ),

  categories: () =>
    fetchAPI<{
      categories: { name: string; count: number }[];
      order: string[];
    }>('/nodes/categories'),

  grouped: () =>
    fetchAPI<{
      categories: string[];
      groups: Record<string, NodeDefinition[]>;
    }>('/nodes/grouped'),
};

/** Pipelines API */
export const pipelinesAPI = {
  validate: (graph: PipelineGraphAPI) =>
    fetchAPI<{
      valid: boolean;
      errors: string[];
      node_count: number;
      edge_count: number;
    }>('/pipelines/validate', {
      method: 'POST',
      body: JSON.stringify(graph),
    }),

  analyze: (graph: PipelineGraphAPI) =>
    fetchAPI<{
      execution_order: string[];
      source_nodes: string[];
      sink_nodes: string[];
      dependencies: Record<string, string[]>;
    }>('/pipelines/analyze', {
      method: 'POST',
      body: JSON.stringify(graph),
    }),

  execute: (graph: PipelineGraphAPI) =>
    fetchAPI<ExecutionResult>('/pipelines/execute', {
      method: 'POST',
      body: JSON.stringify({ graph }),
    }),

  fromTemplate: (templateName: string, params: Record<string, unknown>) =>
    fetchAPI<PipelineGraphAPI>(
      `/pipelines/from-template?template_name=${encodeURIComponent(templateName)}`,
      {
        method: 'POST',
        body: JSON.stringify(params),
      }
    ),
};

/** Presets API */
export const presetsAPI = {
  list: () =>
    fetchAPI<{
      presets: Preset[];
      categories: string[];
    }>('/presets/'),

  get: (presetId: string) => fetchAPI<Preset>(`/presets/${presetId}`),

  getParameters: (presetId: string) =>
    fetchAPI<{
      preset_id: string;
      parameters: Preset['parameters'];
      grouped: Record<string, Preset['parameters']>;
      required_count: number;
    }>(`/presets/${presetId}/parameters`),

  validate: (presetId: string, params: Record<string, unknown>) =>
    fetchAPI<{
      valid: boolean;
      errors: string[];
    }>(`/presets/${presetId}/validate`, {
      method: 'POST',
      body: JSON.stringify(params),
    }),

  buildPipeline: (presetId: string, params: Record<string, unknown>) =>
    fetchAPI<PipelineGraphAPI>(`/presets/${presetId}/build-pipeline`, {
      method: 'POST',
      body: JSON.stringify(params),
    }),
};

/** References API */
export const referencesAPI = {
  listBundled: () =>
    fetchAPI<{
      count: number;
      references: BundledReference[];
      path: string;
      description: string;
    }>('/references/bundled'),

  getBundled: (id: string) =>
    fetchAPI<BundledReference & {
      wavelength_range_nm: [number, number] | null;
      columns: string[];
      sample_lines: Record<string, unknown>[];
    }>(`/references/bundled/${id}`),

  getBundledLines: (
    id: string,
    filters?: {
      species?: string;
      min_wavelength?: number;
      max_wavelength?: number;
      min_intensity?: number;
      limit?: number;
    }
  ) => {
    const params = new URLSearchParams();
    if (filters?.species) params.set('species', filters.species);
    if (filters?.min_wavelength) params.set('min_wavelength', String(filters.min_wavelength));
    if (filters?.max_wavelength) params.set('max_wavelength', String(filters.max_wavelength));
    if (filters?.min_intensity) params.set('min_intensity', String(filters.min_intensity));
    if (filters?.limit) params.set('limit', String(filters.limit));

    const query = params.toString();
    return fetchAPI<{
      reference_id: string;
      line_count: number;
      total_in_file: number;
      lines: Array<{
        wavelength_nm: number;
        species: string;
        intensity: unknown;
      }>;
    }>(`/references/bundled/${id}/lines${query ? `?${query}` : ''}`);
  },
};

/** Visualizations API */
export const visualizationsAPI = {
  signal: (signal: { wavelength: number[]; intensity: number[] }, _options?: { title?: string; normalize?: boolean }) =>
    fetchAPI<VisualizationResponse>('/visualizations/signal', {
      method: 'POST',
      body: JSON.stringify(signal),
    }),

  fromData: (type: string, data: Record<string, unknown>, options?: Record<string, unknown>) =>
    fetchAPI<VisualizationResponse>('/visualizations/from-data', {
      method: 'POST',
      body: JSON.stringify({ type, data, options: options || {} }),
    }),

  plotOptions: () =>
    fetchAPI<{
      visualization_types: Array<{ type: string; description: string; options: string[] }>;
      plot_options: Record<string, string>;
      series_options: Record<string, string>;
    }>('/visualizations/plot-options'),
};

/** Files API */
export const filesAPI = {
  list: (path: string, pattern?: string) => {
    const params = new URLSearchParams({ path });
    if (pattern) params.set('pattern', pattern);
    return fetchAPI<{
      path: string;
      files: Array<{
        name: string;
        path: string;
        size: number;
        is_dir: boolean;
        extension: string | null;
      }>;
      parent: string | null;
    }>(`/files/list?${params}`);
  },

  info: (path: string) =>
    fetchAPI<{
      name: string;
      path: string;
      size: number;
      is_dir: boolean;
      extension: string | null;
    }>(`/files/info?path=${encodeURIComponent(path)}`),

  exists: (path: string) =>
    fetchAPI<{
      path: string;
      exists: boolean;
      is_file: boolean;
      is_dir: boolean;
    }>(`/files/exists?path=${encodeURIComponent(path)}`),

  spectrumDirs: (root: string, maxDepth?: number) => {
    const params = new URLSearchParams({ root });
    if (maxDepth) params.set('max_depth', String(maxDepth));
    return fetchAPI<{
      root: string;
      directories: Array<{
        path: string;
        name: string;
        file_count: number;
      }>;
    }>(`/files/spectrum-dirs?${params}`);
  },
};
