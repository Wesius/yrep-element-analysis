/** Zustand store for UI state management */

import { create } from 'zustand';
import { devtools } from 'zustand/middleware';

type Panel = 'nodes' | 'config' | 'results' | 'presets';

interface UIState {
  // Active panel in sidebar
  activePanel: Panel;

  // Modal states
  isHelpModalOpen: boolean;
  helpNodeId: string | null;

  // Snackbar/notifications
  notification: {
    message: string;
    type: 'info' | 'success' | 'warning' | 'error';
  } | null;

  // Actions
  setActivePanel: (panel: Panel) => void;
  openHelpModal: (nodeId: string) => void;
  closeHelpModal: () => void;
  showNotification: (message: string, type?: 'info' | 'success' | 'warning' | 'error') => void;
  clearNotification: () => void;
}

export const useUIStore = create<UIState>()(
  devtools(
    (set) => ({
      activePanel: 'nodes',
      isHelpModalOpen: false,
      helpNodeId: null,
      notification: null,

      setActivePanel: (panel) => set({ activePanel: panel }),

      openHelpModal: (nodeId) =>
        set({ isHelpModalOpen: true, helpNodeId: nodeId }),

      closeHelpModal: () =>
        set({ isHelpModalOpen: false, helpNodeId: null }),

      showNotification: (message, type = 'info') =>
        set({ notification: { message, type } }),

      clearNotification: () => set({ notification: null }),
    }),
    { name: 'ui-store' }
  )
);
