/** YREP Pipeline Builder - Main Application */

import { ReactFlowProvider } from '@xyflow/react';
import { Header } from './components/layout/Header';
import { Canvas } from './components/layout/Canvas';
import { Sidebar } from './components/layout/Sidebar';
import { HelpModal, Notification } from './components/ui';

function App() {
  return (
    <ReactFlowProvider>
      <div className="h-screen flex flex-col bg-slate-900">
        <Header />
        <main className="flex-1 flex overflow-hidden">
          <Canvas />
          <Sidebar />
        </main>
        <HelpModal />
        <Notification />
      </div>
    </ReactFlowProvider>
  );
}

export default App;
