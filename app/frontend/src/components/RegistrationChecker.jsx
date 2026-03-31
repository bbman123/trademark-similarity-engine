import React, { useState } from 'react';
import { Search, Loader2, AlertCircle, Plus, X, SlidersHorizontal } from 'lucide-react';
import RegistrationResult from './RegistrationResult';
import { API_URL } from '../config';

const RegistrationChecker = () => {
  const [trademarks, setTrademarks] = useState(['', '', '']);
  const [threshold, setThreshold] = useState(0.7);
  const [showSettings, setShowSettings] = useState(false);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const activeCount = trademarks.filter(t => t.trim()).length;

  const handleSubmit = async (e) => {
    e.preventDefault();
    const validMarks = trademarks.filter(t => t.trim());

    if (validMarks.length === 0) {
      setError('Please enter at least one trademark name');
      return;
    }

    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const response = await fetch(`${API_URL}/check-registration`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          trademarks: validMarks,
          threshold,
          top_k: 5,
        }),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => null);
        throw new Error(errData?.detail || `HTTP error ${response.status}`);
      }

      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError(err.message || 'Failed to check registration. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setTrademarks(['', '', '']);
    setResults(null);
    setError(null);
  };

  const updateTrademark = (index, value) => {
    const updated = [...trademarks];
    updated[index] = value;
    setTrademarks(updated);
  };

  const loadExamples = () => {
    setTrademarks(['Awo Tie', 'Thick Plaza', 'Matashi Brace']);
    setResults(null);
    setError(null);
  };

  return (
    <div className="max-w-4xl mx-auto">
      {/* Input Card */}
      <div className="card">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-xl font-bold text-gray-900">Check Trademark Registration</h2>
            <p className="text-sm text-gray-500 mt-1">
              Enter up to 3 trademark names to check against the existing database
            </p>
          </div>
          <button
            onClick={() => setShowSettings(!showSettings)}
            className={`p-2 rounded-lg border transition-colors ${
              showSettings ? 'bg-primary-50 border-primary-300 text-primary-700' : 'border-gray-200 text-gray-500 hover:text-gray-700'
            }`}
          >
            <SlidersHorizontal className="w-5 h-5" />
          </button>
        </div>

        {/* Settings Panel */}
        {showSettings && (
          <div className="mb-6 p-4 bg-gray-50 rounded-lg border border-gray-200">
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Similarity Threshold: <span className="text-primary-600">{(threshold * 100).toFixed(0)}%</span>
            </label>
            <input
              type="range"
              min="0.3"
              max="0.95"
              step="0.05"
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-primary-600"
            />
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>30% (Strict)</span>
              <span>70% (Default)</span>
              <span>95% (Lenient)</span>
            </div>
            <p className="text-xs text-gray-500 mt-2">
              Trademarks with similarity above this threshold will be rejected.
            </p>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          {/* Trademark Input Fields */}
          {trademarks.map((tm, index) => (
            <div key={index} className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-full bg-primary-100 text-primary-700 flex items-center justify-center text-sm font-bold flex-shrink-0">
                {index + 1}
              </div>
              <div className="flex-1">
                <input
                  type="text"
                  value={tm}
                  onChange={(e) => updateTrademark(index, e.target.value)}
                  placeholder={`Trademark name ${index + 1}${index === 0 ? ' (required)' : ' (optional)'}`}
                  className="input-field"
                  maxLength={200}
                />
              </div>
              {tm && (
                <button
                  type="button"
                  onClick={() => updateTrademark(index, '')}
                  className="p-2 text-gray-400 hover:text-gray-600"
                >
                  <X className="w-4 h-4" />
                </button>
              )}
            </div>
          ))}

          {/* Action Buttons */}
          <div className="flex gap-4 pt-2">
            <button
              type="submit"
              disabled={loading || activeCount === 0}
              className="btn-primary flex-1 flex items-center justify-center space-x-2"
            >
              {loading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  <span>Checking Registration...</span>
                </>
              ) : (
                <>
                  <Search className="w-5 h-5" />
                  <span>Check Registration ({activeCount} trademark{activeCount !== 1 ? 's' : ''})</span>
                </>
              )}
            </button>
            <button type="button" onClick={handleClear} className="btn-secondary">
              Clear
            </button>
          </div>
        </form>

        {/* Example Button */}
        <div className="mt-6 pt-6 border-t border-gray-200">
          <p className="text-sm font-semibold text-gray-700 mb-3">Try an example:</p>
          <button
            onClick={loadExamples}
            className="px-4 py-2 text-sm bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg transition-colors border border-gray-300"
          >
            Load sample trademarks: "Awo Tie", "Thick Plaza", "Matashi Brace"
          </button>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="mt-6 p-4 bg-red-50 border-2 border-red-200 rounded-lg flex items-start space-x-3 animate-slide-up">
          <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
          <div>
            <h4 className="font-semibold text-red-900">Error</h4>
            <p className="text-sm text-red-700">{error}</p>
          </div>
        </div>
      )}

      {/* Results Display */}
      {results && (
        <div className="mt-6 space-y-6 animate-slide-up">
          {/* Summary Banner */}
          <div className={`p-4 rounded-lg border-2 ${
            results.summary.rejected === 0
              ? 'bg-green-50 border-green-300'
              : results.summary.approved === 0
              ? 'bg-red-50 border-red-300'
              : 'bg-yellow-50 border-yellow-300'
          }`}>
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-bold text-gray-900">Registration Results</h3>
                <p className="text-sm text-gray-600 mt-1">
                  {results.summary.approved} of {results.summary.total_submitted} trademark{results.summary.total_submitted !== 1 ? 's' : ''} approved
                  {' '}&bull; Threshold: {(results.threshold * 100).toFixed(0)}%
                  {' '}&bull; Database: {results.database_size.toLocaleString()} trademarks
                </p>
              </div>
              <div className="flex gap-2">
                {results.summary.approved > 0 && (
                  <span className="px-3 py-1 bg-green-100 text-green-800 text-sm font-bold rounded-full">
                    {results.summary.approved} Approved
                  </span>
                )}
                {results.summary.rejected > 0 && (
                  <span className="px-3 py-1 bg-red-100 text-red-800 text-sm font-bold rounded-full">
                    {results.summary.rejected} Rejected
                  </span>
                )}
              </div>
            </div>
          </div>

          {/* Individual Results */}
          {results.results.map((result, index) => (
            <RegistrationResult key={index} result={result} index={index} />
          ))}
        </div>
      )}
    </div>
  );
};

export default RegistrationChecker;
