/** Toast notification component */

import { useEffect } from 'react';
import { useUIStore } from '../../store';

const typeStyles: Record<string, { bg: string; border: string; icon: string }> = {
  info: {
    bg: 'bg-blue-500/20',
    border: 'border-blue-500/40',
    icon: 'i',
  },
  success: {
    bg: 'bg-green-500/20',
    border: 'border-green-500/40',
    icon: '\u2713',
  },
  warning: {
    bg: 'bg-yellow-500/20',
    border: 'border-yellow-500/40',
    icon: '!',
  },
  error: {
    bg: 'bg-red-500/20',
    border: 'border-red-500/40',
    icon: '\u2717',
  },
};

const typeTextColors: Record<string, string> = {
  info: 'text-blue-400',
  success: 'text-green-400',
  warning: 'text-yellow-400',
  error: 'text-red-400',
};

export function Notification() {
  const notification = useUIStore((s) => s.notification);
  const clearNotification = useUIStore((s) => s.clearNotification);

  useEffect(() => {
    if (!notification) return;

    // Auto-dismiss non-error notifications after 4 seconds
    if (notification.type !== 'error') {
      const timer = setTimeout(() => {
        clearNotification();
      }, 4000);

      return () => clearTimeout(timer);
    }
  }, [notification, clearNotification]);

  if (!notification) return null;

  const style = typeStyles[notification.type] || typeStyles.info;
  const textColor = typeTextColors[notification.type] || typeTextColors.info;

  return (
    <div className="fixed bottom-4 right-4 z-50 max-w-sm animate-in slide-in-from-right-full duration-300">
      <div
        className={`
          ${style.bg} ${style.border}
          border rounded-lg px-4 py-3 shadow-lg backdrop-blur-sm
          flex items-start gap-3
        `}
      >
        <span className={`${textColor} font-bold text-sm flex-shrink-0 w-5 h-5 flex items-center justify-center rounded-full border ${style.border}`}>
          {style.icon}
        </span>
        <p className="text-slate-200 text-sm flex-1">{notification.message}</p>
        <button
          onClick={clearNotification}
          className="text-slate-400 hover:text-white transition-colors text-lg leading-none flex-shrink-0"
          aria-label="Dismiss"
        >
          &times;
        </button>
      </div>
    </div>
  );
}
