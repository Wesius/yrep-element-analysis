/** Sidebar component with tabs for different panels */

import { useUIStore } from '../../store';
import { NodePalette } from '../panels/NodePalette';
import { ConfigPanel } from '../panels/ConfigPanel';
import { PresetsPanel } from '../panels/PresetsPanel';
import { ResultsPanel } from '../panels/ResultsPanel';

const tabs = [
  { id: 'nodes' as const, label: 'Nodes', icon: '📦' },
  { id: 'config' as const, label: 'Config', icon: '⚙️' },
  { id: 'presets' as const, label: 'Presets', icon: '📋' },
  { id: 'results' as const, label: 'Results', icon: '📊' },
];

export function Sidebar() {
  const activePanel = useUIStore((s) => s.activePanel);
  const setActivePanel = useUIStore((s) => s.setActivePanel);

  return (
    <div className="w-80 bg-slate-800 border-l border-slate-700 flex flex-col">
      {/* Tab bar */}
      <div className="flex border-b border-slate-700">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActivePanel(tab.id)}
            className={`
              flex-1 px-2 py-3 text-xs font-medium transition-colors
              ${activePanel === tab.id
                ? 'text-blue-400 border-b-2 border-blue-400 bg-slate-700/50'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700/30'
              }
            `}
          >
            <span className="block text-lg mb-1">{tab.icon}</span>
            {tab.label}
          </button>
        ))}
      </div>

      {/* Panel content */}
      <div className="flex-1 overflow-hidden">
        {activePanel === 'nodes' && <NodePalette />}
        {activePanel === 'config' && <ConfigPanel />}
        {activePanel === 'presets' && <PresetsPanel />}
        {activePanel === 'results' && <ResultsPanel />}
      </div>
    </div>
  );
}
