/** Configuration panel for editing selected node */

import { usePipelineStore } from '../../store';
import type { ConfigField } from '../../types';

export function ConfigPanel() {
  const selectedNodeId = usePipelineStore((s) => s.selectedNodeId);
  const nodes = usePipelineStore((s) => s.nodes);
  const nodeDefinitions = usePipelineStore((s) => s.nodeDefinitions);
  const updateNodeConfig = usePipelineStore((s) => s.updateNodeConfig);
  const deleteNode = usePipelineStore((s) => s.deleteNode);

  const selectedNode = nodes.find((n) => n.id === selectedNodeId);
  const nodeDef = selectedNode
    ? nodeDefinitions.find((d) => d.identifier === selectedNode.data.identifier)
    : null;

  if (!selectedNode || !nodeDef) {
    return (
      <div className="p-4 text-slate-400 text-center">
        <p className="mb-2">No node selected</p>
        <p className="text-sm">Click on a node to edit its configuration</p>
      </div>
    );
  }

  const handleConfigChange = (fieldName: string, value: unknown) => {
    updateNodeConfig(selectedNodeId!, {
      ...selectedNode.data.config,
      [fieldName]: value,
    });
  };

  const handleDelete = () => {
    if (confirm('Delete this node?')) {
      deleteNode(selectedNodeId!);
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Node header */}
      <div className="p-3 border-b border-slate-700">
        <h3 className="font-medium text-slate-100">{nodeDef.title}</h3>
        <p className="text-xs text-slate-400 mt-1">{nodeDef.description}</p>
      </div>

      {/* Configuration fields */}
      <div className="flex-1 overflow-y-auto p-3">
        {nodeDef.config_fields.length === 0 ? (
          <p className="text-sm text-slate-400">
            This node has no configurable parameters.
          </p>
        ) : (
          <div className="space-y-4">
            {nodeDef.config_fields.map((field) => (
              <ConfigFieldInput
                key={field.name}
                field={field}
                value={selectedNode.data.config[field.name]}
                onChange={(value) => handleConfigChange(field.name, value)}
              />
            ))}
          </div>
        )}
      </div>

      {/* Actions */}
      <div className="p-3 border-t border-slate-700 flex gap-2">
        <button
          onClick={handleDelete}
          className="flex-1 px-3 py-2 bg-red-600 hover:bg-red-500 text-white text-sm rounded transition-colors"
        >
          Delete Node
        </button>
      </div>
    </div>
  );
}

/** Individual config field input */
function ConfigFieldInput({
  field,
  value,
  onChange,
}: {
  field: ConfigField;
  value: unknown;
  onChange: (value: unknown) => void;
}) {
  const inputClasses =
    'w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded text-sm text-slate-100 focus:outline-none focus:border-blue-500';

  return (
    <div>
      <label className="block text-sm font-medium text-slate-200 mb-1">
        {field.label}
        {field.required && <span className="text-red-400 ml-1">*</span>}
      </label>

      {field.description && (
        <p className="text-xs text-slate-400 mb-2">{field.description}</p>
      )}

      {field.type === 'string' && (
        <input
          type="text"
          value={(value as string) || ''}
          onChange={(e) => onChange(e.target.value)}
          className={inputClasses}
          placeholder={field.default as string}
        />
      )}

      {field.type === 'number' && (
        <input
          type="number"
          value={(value as number) ?? ''}
          onChange={(e) => onChange(e.target.valueAsNumber)}
          className={inputClasses}
          placeholder={String(field.default ?? '')}
          step="any"
        />
      )}

      {field.type === 'boolean' && (
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={Boolean(value)}
            onChange={(e) => onChange(e.target.checked)}
            className="w-4 h-4 rounded border-slate-600 bg-slate-700 text-blue-500 focus:ring-blue-500"
          />
          <span className="text-sm text-slate-300">
            {value ? 'Enabled' : 'Disabled'}
          </span>
        </label>
      )}

      {field.type === 'select' && field.options && (
        <select
          value={(value as string) || ''}
          onChange={(e) => onChange(e.target.value)}
          className={inputClasses}
        >
          <option value="">Select...</option>
          {field.options.map((opt) => (
            <option key={opt} value={opt}>
              {opt}
            </option>
          ))}
        </select>
      )}

      {(field.type === 'file' || field.type === 'directory') && (
        <input
          type="text"
          value={(value as string) || ''}
          onChange={(e) => onChange(e.target.value)}
          className={inputClasses}
          placeholder={`Enter ${field.type} path...`}
        />
      )}
    </div>
  );
}
