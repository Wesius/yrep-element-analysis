/** Help modal displaying node documentation */

import { useUIStore, usePipelineStore } from '../../store';

export function HelpModal() {
  const isOpen = useUIStore((s) => s.isHelpModalOpen);
  const nodeId = useUIStore((s) => s.helpNodeId);
  const closeHelpModal = useUIStore((s) => s.closeHelpModal);
  const nodeDefinitions = usePipelineStore((s) => s.nodeDefinitions);

  if (!isOpen || !nodeId) return null;

  const nodeDef = nodeDefinitions.find((n) => n.identifier === nodeId);

  if (!nodeDef) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center">
        <div
          className="absolute inset-0 bg-black/60"
          onClick={closeHelpModal}
        />
        <div className="relative bg-slate-800 border border-slate-700 rounded-lg p-6 max-w-md w-full mx-4">
          <p className="text-slate-300">Node documentation not found.</p>
          <button
            onClick={closeHelpModal}
            className="mt-4 px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div
        className="absolute inset-0 bg-black/60"
        onClick={closeHelpModal}
      />
      <div className="relative bg-slate-800 border border-slate-700 rounded-lg max-w-lg w-full mx-4 max-h-[80vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-700">
          <h2 className="text-lg font-semibold text-white">{nodeDef.title}</h2>
          <button
            onClick={closeHelpModal}
            className="text-slate-400 hover:text-white transition-colors text-xl leading-none"
            aria-label="Close"
          >
            &times;
          </button>
        </div>

        {/* Content */}
        <div className="px-6 py-4 overflow-y-auto flex-1">
          <div className="mb-4">
            <span className="inline-block px-2 py-1 text-xs rounded bg-slate-700 text-slate-300">
              {nodeDef.category}
            </span>
          </div>

          <div className="mb-4">
            <h3 className="text-sm font-medium text-slate-400 mb-1">Description</h3>
            <p className="text-slate-200">{nodeDef.description}</p>
          </div>

          {nodeDef.explanation && (
            <div className="mb-4">
              <h3 className="text-sm font-medium text-slate-400 mb-1">How it works</h3>
              <p className="text-slate-300 text-sm">{nodeDef.explanation}</p>
            </div>
          )}

          {nodeDef.tips && nodeDef.tips.length > 0 && (
            <div className="mb-4">
              <h3 className="text-sm font-medium text-slate-400 mb-2">Tips</h3>
              <ul className="space-y-2">
                {nodeDef.tips.map((tip, index) => (
                  <li key={index} className="flex items-start gap-2 text-sm text-slate-300">
                    <span className="text-blue-400 mt-0.5">•</span>
                    <span>{tip}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {(nodeDef.inputs.length > 0 || nodeDef.outputs.length > 0) && (
            <div className="border-t border-slate-700 pt-4 mt-4">
              {nodeDef.inputs.length > 0 && (
                <div className="mb-3">
                  <h3 className="text-sm font-medium text-slate-400 mb-2">Inputs</h3>
                  <div className="space-y-1">
                    {nodeDef.inputs.map((input) => (
                      <div key={input.name} className="text-sm">
                        <span className="text-blue-400">{input.name}</span>
                        <span className="text-slate-500"> ({input.type})</span>
                        {input.required && <span className="text-red-400 ml-1">*</span>}
                        {input.description && (
                          <p className="text-slate-400 text-xs ml-2">{input.description}</p>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {nodeDef.outputs.length > 0 && (
                <div>
                  <h3 className="text-sm font-medium text-slate-400 mb-2">Outputs</h3>
                  <div className="space-y-1">
                    {nodeDef.outputs.map((output) => (
                      <div key={output.name} className="text-sm">
                        <span className="text-green-400">{output.name}</span>
                        <span className="text-slate-500"> ({output.type})</span>
                        {output.description && (
                          <p className="text-slate-400 text-xs ml-2">{output.description}</p>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-3 border-t border-slate-700">
          <button
            onClick={closeHelpModal}
            className="w-full px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
