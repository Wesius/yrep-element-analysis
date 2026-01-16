/** Presets panel for quick-start workflow templates */

import { useState, useEffect } from 'react';
import { presetsAPI, referencesAPI } from '../../api/client';
import type { Preset, PresetParameter, BundledReference } from '../../types';
import { usePipelineStore, useUIStore } from '../../store';

/** Icon mapping for preset categories */
const presetIcons: Record<string, string> = {
  Detection: '🔍',
  Processing: '⚙️',
  Analysis: '📊',
  default: '📋',
};

export function PresetsPanel() {
  const [presets, setPresets] = useState<Preset[]>([]);
  const [selectedPreset, setSelectedPreset] = useState<Preset | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Load presets on mount
  useEffect(() => {
    const loadPresets = async () => {
      try {
        const result = await presetsAPI.list();
        setPresets(result.presets);
      } catch (err) {
        setError(`Failed to load presets: ${err}`);
      } finally {
        setLoading(false);
      }
    };

    loadPresets();
  }, []);

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
        {presets.map((preset) => (
          <PresetCard
            key={preset.id}
            preset={preset}
            onSelect={() => setSelectedPreset(preset)}
          />
        ))}
      </div>
    </div>
  );
}

/** Individual preset card */
function PresetCard({
  preset,
  onSelect,
}: {
  preset: Preset;
  onSelect: () => void;
}) {
  const icon = presetIcons[preset.category] || presetIcons.default;

  return (
    <button
      onClick={onSelect}
      className="w-full text-left p-3 bg-slate-700 rounded-lg hover:bg-slate-600 transition-colors"
    >
      <div className="flex items-start gap-3">
        <span className="text-2xl">{icon}</span>
        <div className="flex-1 min-w-0">
          <h4 className="font-medium text-slate-100">{preset.name}</h4>
          <p className="text-xs text-slate-400 mt-1 line-clamp-2">
            {preset.description}
          </p>
          <div className="flex flex-wrap gap-1 mt-2">
            {preset.use_cases.slice(0, 2).map((useCase, i) => (
              <span
                key={i}
                className="text-xs px-2 py-0.5 bg-slate-600 rounded text-slate-300"
              >
                {useCase}
              </span>
            ))}
          </div>
        </div>
        <span className="text-slate-500">→</span>
      </div>
    </button>
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
    const loadRefs = async () => {
      try {
        const result = await referencesAPI.listBundled();
        setBundledRefs(result.references);
      } catch {
        // Ignore - refs are optional
      }
    };
    loadRefs();
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
      setErrors([`Validation failed: ${err}`]);
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
      setErrors([`Failed to build pipeline: ${err}`]);
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
      setErrors([`Execution failed: ${err}`]);
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
              {errors.map((err, i) => (
                <li key={i}>• {err}</li>
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
          disabled={validating}
          className="w-full px-4 py-2 text-sm text-slate-300 hover:text-white hover:bg-slate-700 rounded transition-colors"
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

      {(parameter.type === 'string' || parameter.type === 'file' || parameter.type === 'directory') && (
        <>
          <input
            type="text"
            value={(value as string) || ''}
            onChange={(e) => onChange(e.target.value)}
            className={inputClasses}
            placeholder={
              parameter.type === 'file' ? 'Enter file path...' :
              parameter.type === 'directory' ? 'Enter directory path...' :
              ''
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
                      const dirPath = ref.path.substring(0, ref.path.lastIndexOf('/'));
                      onChange(dirPath);
                    }
                  }
                }}
                className="w-full px-2 py-1 bg-slate-600 border border-slate-500 rounded text-xs text-slate-300"
                defaultValue=""
              >
                <option value="">Select bundled references...</option>
                {bundledRefs.slice(0, 1).map((ref) => (
                  <option key={ref.id} value={ref.id}>
                    Bundled: {ref.path.substring(0, ref.path.lastIndexOf('/'))}
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
