/** Presets panel for quick-start workflow templates */

import { useState, useEffect } from 'react';
import { presetsAPI, referencesAPI } from '../../api/client';
import type { Preset, PresetParameter, BundledReference } from '../../types';
import { usePipelineStore, useUIStore } from '../../store';
import { SpectraDropZone } from '../common/SpectraDropZone';

/** Extract error message from unknown error */
function getErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
}

/** Safely extract directory path from a file path */
function getDirectoryPath(filePath: string): string {
  const lastSlash = filePath.lastIndexOf('/');
  if (lastSlash === -1) {
    return filePath; // No slash found, return as-is
  }
  return filePath.substring(0, lastSlash);
}


export function PresetsPanel() {
  const [presets, setPresets] = useState<Preset[]>([]);
  const [selectedPreset, setSelectedPreset] = useState<Preset | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [buildingPreset, setBuildingPreset] = useState<string | null>(null);

  const loadPipeline = usePipelineStore((s) => s.loadPipeline);
  const setActivePanel = useUIStore((s) => s.setActivePanel);

  // Load presets on mount
  useEffect(() => {
    const controller = new AbortController();

    const loadPresets = async () => {
      try {
        const result = await presetsAPI.list();
        if (!controller.signal.aborted) {
          setPresets(result.presets);
        }
      } catch (err) {
        if (!controller.signal.aborted) {
          setError(`Failed to load presets: ${getErrorMessage(err)}`);
        }
      } finally {
        if (!controller.signal.aborted) {
          setLoading(false);
        }
      }
    };

    loadPresets();

    return () => {
      controller.abort();
    };
  }, []);

  /** Build pipeline from preset with default parameters */
  const handleBuildPreset = async (preset: Preset) => {
    setBuildingPreset(preset.id);
    try {
      // Build default parameters from preset definition
      const defaults: Record<string, unknown> = {};
      preset.parameters.forEach((p) => {
        defaults[p.name] = p.default ?? '';
      });

      const pipeline = await presetsAPI.buildPipeline(preset.id, defaults);

      // Convert API pipeline to React Flow nodes/edges
      const nodes = pipeline.nodes.map((n, i) => ({
        id: n.id,
        type: 'pipeline',
        position: n.position || { x: 200 * (i % 3), y: 150 * Math.floor(i / 3) },
        data: {
          identifier: n.identifier,
          title: n.identifier.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase()),
          category: 'Preset',
          config: n.config,
          inputs: [],
          outputs: [],
        },
      }));

      const edges = pipeline.edges.map((e) => ({
        id: e.id,
        source: e.source_node,
        target: e.target_node,
        sourceHandle: `output-${e.source_port}`,
        targetHandle: `input-${e.target_port}`,
        type: 'smoothstep',
      }));

      loadPipeline(nodes, edges);
      setActivePanel('nodes');
    } catch (err) {
      setError(`Failed to build pipeline: ${getErrorMessage(err)}`);
    } finally {
      setBuildingPreset(null);
    }
  };

  if (loading) {
    return (
      <div className="p-4 text-slate-400 text-center">
        Loading presets...
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 text-red-400 text-center">
        <p className="mb-2">Error</p>
        <p className="text-sm">{error}</p>
      </div>
    );
  }

  // If a preset is selected, show parameter form
  if (selectedPreset) {
    return (
      <PresetParameterForm
        preset={selectedPreset}
        onBack={() => setSelectedPreset(null)}
      />
    );
  }

  // Show preset list
  return (
    <div className="flex flex-col h-full">
      <div className="p-3 border-b border-slate-700">
        <h3 className="font-medium text-slate-100">Quick Start Presets</h3>
        <p className="text-xs text-slate-400 mt-1">
          Pre-configured workflows for common tasks
        </p>
      </div>

      <div className="flex-1 overflow-y-auto p-3 space-y-3">
        {presets.length === 0 ? (
          <div className="text-slate-400 text-center py-8">
            <p className="mb-2">No presets available</p>
            <p className="text-sm">Check your backend configuration</p>
          </div>
        ) : (
          presets.map((preset) => (
            <PresetCard
              key={preset.id}
              preset={preset}
              onBuild={() => handleBuildPreset(preset)}
              onConfigure={() => setSelectedPreset(preset)}
              isBuilding={buildingPreset === preset.id}
            />
          ))
        )}
      </div>
    </div>
  );
}

/** Individual preset card - clicking anywhere builds graph on canvas */
function PresetCard({
  preset,
  onBuild,
  onConfigure,
  isBuilding,
}: {
  preset: Preset;
  onBuild: () => void;
  onConfigure: () => void;
  isBuilding: boolean;
}) {
  return (
    <div
      onClick={isBuilding ? undefined : onBuild}
      className={`
        w-full p-3 bg-slate-700 rounded-lg transition-colors
        ${isBuilding ? 'cursor-not-allowed opacity-75' : 'cursor-pointer hover:bg-slate-600'}
      `}
    >
      <div className="flex items-start gap-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between">
            <h4 className="font-medium text-slate-100">{preset.name}</h4>
            <button
              onClick={(e) => {
                e.stopPropagation();
                onConfigure();
              }}
              disabled={isBuilding}
              className="px-2 py-1 text-sm text-slate-400 hover:text-white hover:bg-slate-500 rounded transition-colors"
              title="Configure parameters"
            >
              Configure
            </button>
          </div>
          <p className="text-xs text-slate-400 mt-1 line-clamp-2">
            {preset.description}
          </p>
          <div className="flex flex-wrap gap-1 mt-2">
            {preset.use_cases.slice(0, 2).map((useCase) => (
              <span
                key={useCase}
                className="text-xs px-2 py-0.5 bg-slate-600 rounded text-slate-300"
              >
                {useCase}
              </span>
            ))}
          </div>
          {isBuilding && (
            <p className="text-xs text-blue-400 mt-2">Building graph...</p>
          )}
        </div>
      </div>
    </div>
  );
}

/** Parameter form for a preset */
function PresetParameterForm({
  preset,
  onBack,
}: {
  preset: Preset;
  onBack: () => void;
}) {
  const [parameters, setParameters] = useState<Record<string, unknown>>(() => {
    // Initialize with defaults
    const defaults: Record<string, unknown> = {};
    preset.parameters.forEach((p) => {
      defaults[p.name] = p.default ?? '';
    });
    return defaults;
  });
  const [bundledRefs, setBundledRefs] = useState<BundledReference[]>([]);
  const [validating, setValidating] = useState(false);
  const [executing, setExecuting] = useState(false);
  const [errors, setErrors] = useState<string[]>([]);

  const loadPipeline = usePipelineStore((s) => s.loadPipeline);
  const setExecutionResult = usePipelineStore((s) => s.setExecutionResult);
  const setActivePanel = useUIStore((s) => s.setActivePanel);

  // Load bundled references for reference_path fields
  useEffect(() => {
    const controller = new AbortController();

    const loadRefs = async () => {
      try {
        const result = await referencesAPI.listBundled();
        if (!controller.signal.aborted) {
          setBundledRefs(result.references);
        }
      } catch {
        // Ignore - refs are optional
      }
    };
    loadRefs();

    return () => {
      controller.abort();
    };
  }, []);

  const handleParameterChange = (name: string, value: unknown) => {
    setParameters((prev) => ({ ...prev, [name]: value }));
    setErrors([]);
  };

  const handleValidate = async () => {
    setValidating(true);
    try {
      const result = await presetsAPI.validate(preset.id, parameters);
      if (result.valid) {
        setErrors([]);
        return true;
      } else {
        setErrors(result.errors);
        return false;
      }
    } catch (err) {
      setErrors([`Validation failed: ${getErrorMessage(err)}`]);
      return false;
    } finally {
      setValidating(false);
    }
  };

  const handleBuildPipeline = async () => {
    const valid = await handleValidate();
    if (!valid) return;

    try {
      const pipeline = await presetsAPI.buildPipeline(preset.id, parameters);

      // Convert API pipeline to React Flow nodes/edges
      const nodes = pipeline.nodes.map((n, i) => ({
        id: n.id,
        type: 'pipeline',
        position: n.position || { x: 200 * (i % 3), y: 150 * Math.floor(i / 3) },
        data: {
          identifier: n.identifier,
          title: n.identifier.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase()),
          category: 'Preset',
          config: n.config,
          inputs: [],
          outputs: [],
        },
      }));

      const edges = pipeline.edges.map((e) => ({
        id: e.id,
        source: e.source_node,
        target: e.target_node,
        sourceHandle: `output-${e.source_port}`,
        targetHandle: `input-${e.target_port}`,
        type: 'smoothstep',
      }));

      loadPipeline(nodes, edges);
      setActivePanel('nodes');
    } catch (err) {
      setErrors([`Failed to build pipeline: ${getErrorMessage(err)}`]);
    }
  };

  const handleExecute = async () => {
    const valid = await handleValidate();
    if (!valid) return;

    setExecuting(true);
    try {
      // Build and execute in one step
      const pipeline = await presetsAPI.buildPipeline(preset.id, parameters);

      // Execute the pipeline
      const { pipelinesAPI } = await import('../../api/client');
      const result = await pipelinesAPI.execute(pipeline);

      setExecutionResult(result);
      setActivePanel('results');
    } catch (err) {
      setErrors([`Execution failed: ${getErrorMessage(err)}`]);
    } finally {
      setExecuting(false);
    }
  };

  // Group parameters
  const groupedParams = preset.parameters.reduce((acc, param) => {
    const group = param.group || 'General';
    if (!acc[group]) acc[group] = [];
    acc[group].push(param);
    return acc;
  }, {} as Record<string, PresetParameter[]>);

  return (
    <div className="flex flex-col h-full">
      {/* Header with back button */}
      <div className="p-3 border-b border-slate-700">
        <button
          onClick={onBack}
          className="text-sm text-slate-400 hover:text-slate-200 mb-2"
        >
          ← Back to presets
        </button>
        <h3 className="font-medium text-slate-100">{preset.name}</h3>
        <p className="text-xs text-slate-400 mt-1">{preset.description}</p>
      </div>

      {/* Parameter form */}
      <div className="flex-1 overflow-y-auto p-3">
        {Object.entries(groupedParams).map(([group, params]) => (
          <div key={group} className="mb-4">
            <h4 className="text-xs font-medium text-slate-400 uppercase tracking-wide mb-2">
              {group}
            </h4>
            <div className="space-y-3">
              {params.map((param) => (
                <ParameterInput
                  key={param.name}
                  parameter={param}
                  value={parameters[param.name]}
                  onChange={(value) => handleParameterChange(param.name, value)}
                  bundledRefs={bundledRefs}
                />
              ))}
            </div>
          </div>
        ))}

        {/* Errors */}
        {errors.length > 0 && (
          <div className="mt-4 p-3 bg-red-900/30 border border-red-700 rounded">
            <h4 className="text-sm font-medium text-red-400 mb-1">
              Validation Errors
            </h4>
            <ul className="text-xs text-red-300 space-y-1">
              {errors.map((err) => (
                <li key={err}>• {err}</li>
              ))}
            </ul>
          </div>
        )}

        {/* Explanation */}
        <details className="mt-4">
          <summary className="text-sm text-slate-400 cursor-pointer hover:text-slate-300">
            How this preset works
          </summary>
          <div className="mt-2 text-xs text-slate-400 whitespace-pre-wrap bg-slate-700/50 p-3 rounded">
            {preset.explanation}
          </div>
        </details>
      </div>

      {/* Actions */}
      <div className="p-3 border-t border-slate-700 space-y-2">
        <button
          onClick={handleExecute}
          disabled={executing}
          className={`
            w-full px-4 py-2 text-sm font-medium rounded transition-colors
            ${executing
              ? 'bg-slate-600 text-slate-400 cursor-not-allowed'
              : 'bg-blue-600 hover:bg-blue-500 text-white'
            }
          `}
        >
          {executing ? 'Executing...' : 'Run Analysis'}
        </button>
        <button
          onClick={handleBuildPipeline}
          disabled={validating || executing}
          className={`
            w-full px-4 py-2 text-sm rounded transition-colors
            ${validating || executing
              ? 'text-slate-500 cursor-not-allowed'
              : 'text-slate-300 hover:text-white hover:bg-slate-700'
            }
          `}
        >
          Build Pipeline (Edit First)
        </button>
      </div>
    </div>
  );
}

/** Individual parameter input */
function ParameterInput({
  parameter,
  value,
  onChange,
  bundledRefs,
}: {
  parameter: PresetParameter;
  value: unknown;
  onChange: (value: unknown) => void;
  bundledRefs: BundledReference[];
}) {
  const inputClasses =
    'w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded text-sm text-slate-100 focus:outline-none focus:border-blue-500';

  // Check if this is a reference path field
  const isReferencePath = parameter.name.includes('reference') ||
    parameter.label.toLowerCase().includes('reference');

  return (
    <div>
      <label className="block text-sm font-medium text-slate-200 mb-1">
        {parameter.label}
        {parameter.required && <span className="text-red-400 ml-1">*</span>}
      </label>

      {parameter.description && (
        <p className="text-xs text-slate-400 mb-2">{parameter.description}</p>
      )}

      {parameter.type === 'number' && (
        <input
          type="number"
          value={(value as number) ?? ''}
          onChange={(e) => onChange(e.target.valueAsNumber)}
          className={inputClasses}
          step="any"
        />
      )}

      {parameter.type === 'boolean' && (
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={Boolean(value)}
            onChange={(e) => onChange(e.target.checked)}
            className="w-4 h-4 rounded border-slate-600 bg-slate-700 text-blue-500"
          />
          <span className="text-sm text-slate-300">
            {value ? 'Enabled' : 'Disabled'}
          </span>
        </label>
      )}

      {parameter.type === 'string' && (
        <input
          type="text"
          value={(value as string) || ''}
          onChange={(e) => onChange(e.target.value)}
          className={inputClasses}
        />
      )}

      {(parameter.type === 'file' || parameter.type === 'directory') && (
        <>
          <SpectraDropZone
            value={(value as string) || ''}
            onChange={onChange}
            mode={parameter.type}
            placeholder={
              parameter.type === 'file'
                ? 'Drop spectrum file here or click to upload'
                : 'Drop spectrum files here or click to upload'
            }
          />

          {/* Quick select for reference paths */}
          {isReferencePath && bundledRefs.length > 0 && (
            <div className="mt-2">
              <p className="text-xs text-slate-500 mb-1">Or use bundled:</p>
              <select
                onChange={(e) => {
                  if (e.target.value) {
                    const ref = bundledRefs.find(r => r.id === e.target.value);
                    if (ref) {
                      // Use the directory containing the bundled refs
                      const dirPath = getDirectoryPath(ref.path);
                      onChange(dirPath);
                    }
                  }
                }}
                className="w-full px-2 py-1 bg-slate-600 border border-slate-500 rounded text-xs text-slate-300"
                defaultValue=""
              >
                <option value="">Select bundled references...</option>
                {bundledRefs.map((ref) => (
                  <option key={ref.id} value={ref.id}>
                    {ref.name}
                  </option>
                ))}
              </select>
            </div>
          )}
        </>
      )}
    </div>
  );
}
