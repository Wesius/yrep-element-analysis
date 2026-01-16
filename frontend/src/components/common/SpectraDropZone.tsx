/** Drag-and-drop upload zone for spectrum files */

import { useState, useCallback, useRef, useEffect } from 'react';
import { filesAPI } from '../../api/client';

interface SpectraDropZoneProps {
  /** Current file path value */
  value: string;
  /** Called when a file is selected or uploaded */
  onChange: (path: string) => void;
  /** Placeholder text for empty state */
  placeholder?: string;
  /** Whether this is for a single file or directory */
  mode?: 'file' | 'directory';
}

interface UploadedFileInfo {
  name: string;
  path: string;
  size: number;
  is_dir: boolean;
}

export function SpectraDropZone({
  value,
  onChange,
  placeholder = 'Drop spectrum file here or click to browse',
  mode = 'file',
}: SpectraDropZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [recentUploads, setRecentUploads] = useState<UploadedFileInfo[]>([]);
  const [showRecent, setShowRecent] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Load recent uploads on mount
  useEffect(() => {
    const loadRecentUploads = async () => {
      try {
        const result = await filesAPI.listUploaded();
        setRecentUploads(result.files.slice(0, 5));
      } catch {
        // Ignore - uploads may not exist yet
      }
    };
    loadRecentUploads();
  }, []);

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleUpload = useCallback(
    async (files: File[]) => {
      if (files.length === 0) return;

      setIsUploading(true);
      setUploadError(null);

      try {
        const result = await filesAPI.upload(files);

        if (result.uploaded.length > 0) {
          // Use the first uploaded file's path
          const uploadedPath =
            mode === 'directory'
              ? result.upload_directory
              : result.uploaded[0].path;
          onChange(uploadedPath);

          // Refresh recent uploads
          const updatedList = await filesAPI.listUploaded();
          setRecentUploads(updatedList.files.slice(0, 5));
        }

        if (result.errors.length > 0) {
          setUploadError(result.errors.join('; '));
        }
      } catch (err) {
        setUploadError(err instanceof Error ? err.message : 'Upload failed');
      } finally {
        setIsUploading(false);
      }
    },
    [mode, onChange]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(false);

      const files = Array.from(e.dataTransfer.files);
      handleUpload(files);
    },
    [handleUpload]
  );

  const handleFileInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = e.target.files ? Array.from(e.target.files) : [];
      handleUpload(files);
      // Reset input so same file can be selected again
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    },
    [handleUpload]
  );

  const handleClick = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const handleManualInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      onChange(e.target.value);
      setUploadError(null);
    },
    [onChange]
  );

  const handleSelectRecent = useCallback(
    (path: string) => {
      onChange(path);
      setShowRecent(false);
    },
    [onChange]
  );

  const filename = value ? value.split('/').pop() : null;

  return (
    <div className="space-y-2">
      {/* Drop zone */}
      <div
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        onClick={handleClick}
        className={`
          relative border-2 border-dashed rounded-lg p-4 text-center cursor-pointer
          transition-colors duration-200
          ${
            isDragging
              ? 'border-blue-400 bg-blue-900/20'
              : 'border-slate-600 hover:border-slate-500 hover:bg-slate-700/50'
          }
          ${isUploading ? 'opacity-50 pointer-events-none' : ''}
        `}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".txt"
          multiple={mode === 'directory'}
          onChange={handleFileInputChange}
          className="hidden"
        />

        {isUploading ? (
          <div className="flex items-center justify-center gap-2 text-slate-400">
            <svg
              className="animate-spin h-5 w-5"
              fill="none"
              viewBox="0 0 24 24"
            >
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
              />
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
              />
            </svg>
            <span>Uploading...</span>
          </div>
        ) : value ? (
          <div className="text-sm">
            <div className="text-slate-200 font-medium truncate">{filename}</div>
            <div className="text-slate-400 text-xs mt-1 truncate">{value}</div>
          </div>
        ) : (
          <div className="text-slate-400 text-sm">
            <div className="mb-1">{placeholder}</div>
            <div className="text-xs text-slate-500">
              Supports .txt spectrum files
            </div>
          </div>
        )}
      </div>

      {/* Error message */}
      {uploadError && (
        <div className="text-xs text-red-400 bg-red-900/20 px-2 py-1 rounded">
          {uploadError}
        </div>
      )}

      {/* Manual path input */}
      <div className="flex gap-2">
        <input
          type="text"
          value={value}
          onChange={handleManualInput}
          placeholder="Or enter path manually..."
          className="flex-1 px-3 py-1.5 bg-slate-700 border border-slate-600 rounded text-sm text-slate-100 focus:outline-none focus:border-blue-500"
        />
        {recentUploads.length > 0 && (
          <button
            type="button"
            onClick={() => setShowRecent(!showRecent)}
            className="px-2 py-1.5 bg-slate-600 hover:bg-slate-500 rounded text-xs text-slate-300 transition-colors"
            title="Recent uploads"
          >
            Recent
          </button>
        )}
      </div>

      {/* Recent uploads dropdown */}
      {showRecent && recentUploads.length > 0 && (
        <div className="bg-slate-700 border border-slate-600 rounded overflow-hidden">
          <div className="text-xs text-slate-400 px-2 py-1 border-b border-slate-600">
            Recent uploads
          </div>
          {recentUploads.map((file) => (
            <button
              key={file.path}
              type="button"
              onClick={() => handleSelectRecent(file.path)}
              className="w-full text-left px-2 py-1.5 text-sm text-slate-200 hover:bg-slate-600 transition-colors truncate"
            >
              {file.name}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
