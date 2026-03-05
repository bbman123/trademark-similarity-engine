import React from 'react';
import { CheckCircle2, XCircle, Loader2, RefreshCw } from 'lucide-react';

const ApiStatus = ({ status, onRetry }) => {
  if (status.status === 'checking') {
    return (
      <div className="inline-flex items-center gap-2 px-4 py-2 bg-blue-50 border border-blue-200 rounded-lg">
        <Loader2 className="w-4 h-4 animate-spin text-blue-600" />
        <span className="text-sm font-medium text-blue-700">Checking API status...</span>
      </div>
    );
  }

  if (status.status === 'error') {
    return (
      <div className="inline-flex items-center gap-3 px-4 py-2 bg-red-50 border border-red-200 rounded-lg">
        <XCircle className="w-4 h-4 text-red-600" />
        <span className="text-sm font-medium text-red-700">API unavailable</span>
        <button
          onClick={onRetry}
          className="ml-2 p-1 hover:bg-red-100 rounded transition-colors"
          title="Retry connection"
        >
          <RefreshCw className="w-3.5 h-3.5 text-red-600" />
        </button>
      </div>
    );
  }

  return (
    <div className="inline-flex items-center gap-2 px-4 py-2 bg-green-50 border border-green-200 rounded-lg">
      <CheckCircle2 className="w-4 h-4 text-green-600" />
      <span className="text-sm font-medium text-green-700">
        API Online • Models Loaded
      </span>
    </div>
  );
};

export default ApiStatus;
