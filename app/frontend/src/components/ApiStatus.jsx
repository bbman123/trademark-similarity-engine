import React from 'react';
import { CheckCircle2, AlertCircle, Loader2, RefreshCw } from 'lucide-react';

const ApiStatus = ({ status, onRetry }) => {
  if (status.status === 'checking') {
    return (
      <div className="inline-flex items-center gap-2 px-4 py-2 bg-blue-50 border border-blue-200 rounded-full text-sm text-blue-700">
        <Loader2 className="w-4 h-4 animate-spin" />
        <span>Connecting to API...</span>
      </div>
    );
  }

  if (status.status === 'error') {
    return (
      <div className="inline-flex items-center gap-2 px-4 py-2 bg-red-50 border border-red-200 rounded-full text-sm text-red-700">
        <AlertCircle className="w-4 h-4" />
        <span>API unavailable</span>
        <button onClick={onRetry} className="ml-1 hover:text-red-900">
          <RefreshCw className="w-3.5 h-3.5" />
        </button>
      </div>
    );
  }

  const dbSize = status.data?.database_size;
  return (
    <div className="inline-flex items-center gap-2 px-4 py-2 bg-green-50 border border-green-200 rounded-full text-sm text-green-700">
      <CheckCircle2 className="w-4 h-4" />
      <span>System Online{dbSize ? ` \u2022 ${dbSize.toLocaleString()} trademarks in database` : ''}</span>
    </div>
  );
};

export default ApiStatus;
