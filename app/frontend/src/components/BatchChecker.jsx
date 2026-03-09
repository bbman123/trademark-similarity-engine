import React, { useState } from 'react';
import { Plus, Trash2, Upload, Loader2, AlertCircle, Download } from 'lucide-react';
import { API_URL } from '../config';

const BatchChecker = () => {
  const [pairs, setPairs] = useState([
    { id: 1, mark1: '', mark2: '' },
    { id: 2, mark1: '', mark2: '' },
  ]);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const addPair = () => {
    if (pairs.length >= 100) {
      setError('Maximum 100 pairs allowed');
      return;
    }
    setPairs([...pairs, { id: Date.now(), mark1: '', mark2: '' }]);
  };

  const removePair = (id) => {
    if (pairs.length <= 1) return;
    setPairs(pairs.filter(pair => pair.id !== id));
  };

  const updatePair = (id, field, value) => {
    setPairs(pairs.map(pair =>
      pair.id === id ? { ...pair, [field]: value } : pair
    ));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    const validPairs = pairs.filter(p => p.mark1.trim() && p.mark2.trim());
    
    if (validPairs.length === 0) {
      setError('Please enter at least one valid trademark pair');
      return;
    }

    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const response = await fetch(`${API_URL}/batch-similarity`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          pairs: validPairs,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError(err.message || 'Failed to process batch. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setPairs([
      { id: 1, mark1: '', mark2: '' },
      { id: 2, mark1: '', mark2: '' },
    ]);
    setResults(null);
    setError(null);
  };

  const loadExamples = () => {
    setPairs([
      { id: 1, mark1: 'BrandA', mark2: 'BrandB' },
      { id: 2, mark1: 'CoffeePlus', mark2: 'PlusCoffee' },
      { id: 3, mark1: 'TechSmart', mark2: 'TechGenius' },
      { id: 4, mark1: 'SuperMarket', mark2: 'MarketPro' },
      { id: 5, mark1: 'GlobalTrade', mark2: 'TradeGlobal' },
    ]);
    setResults(null);
    setError(null);
  };

  const exportResults = () => {
    if (!results) return;
    
    const csv = [
      ['Mark 1', 'Mark 2', 'Result', 'Probability', 'Risk Level'],
      ...results.results.map(r => [
        r.mark1,
        r.mark2,
        r.label_text,
        `${(r.probability * 100).toFixed(1)}%`,
        r.risk_level
      ])
    ].map(row => row.join(',')).join('\n');

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `trademark-similarity-results-${Date.now()}.csv`;
    a.click();
  };

  const getRiskBadgeClass = (risk) => {
    switch (risk) {
      case 'HIGH':
        return 'badge-high';
      case 'MEDIUM':
        return 'badge-medium';
      case 'LOW':
        return 'badge-low';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="max-w-6xl mx-auto">
      {/* Main Card */}
      <div className="card">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h3 className="text-xl font-bold text-gray-900">Batch Trademark Check</h3>
            <p className="text-sm text-gray-600 mt-1">
              Check multiple trademark pairs at once (up to 100 pairs)
            </p>
          </div>
          <button
            type="button"
            onClick={loadExamples}
            className="btn-secondary flex items-center gap-2"
          >
            <Upload className="w-4 h-4" />
            Load Examples
          </button>
        </div>

        <form onSubmit={handleSubmit}>
          {/* Pair Inputs */}
          <div className="space-y-3 mb-6 max-h-96 overflow-y-auto pr-2">
            {pairs.map((pair, index) => (
              <div
                key={pair.id}
                className="flex items-center gap-3 p-4 bg-gray-50 rounded-lg border border-gray-200 hover:border-gray-300 transition-colors"
              >
                <span className="text-sm font-semibold text-gray-500 w-8">#{index + 1}</span>
                
                <input
                  type="text"
                  value={pair.mark1}
                  onChange={(e) => updatePair(pair.id, 'mark1', e.target.value)}
                  placeholder="First trademark"
                  className="flex-1 px-3 py-2 rounded-lg border border-gray-300 focus:border-primary-500 focus:ring-1 focus:ring-primary-500 outline-none text-sm"
                  maxLength={200}
                />
                
                <span className="text-gray-400 text-sm">vs</span>
                
                <input
                  type="text"
                  value={pair.mark2}
                  onChange={(e) => updatePair(pair.id, 'mark2', e.target.value)}
                  placeholder="Second trademark"
                  className="flex-1 px-3 py-2 rounded-lg border border-gray-300 focus:border-primary-500 focus:ring-1 focus:ring-primary-500 outline-none text-sm"
                  maxLength={200}
                />
                
                <button
                  type="button"
                  onClick={() => removePair(pair.id)}
                  disabled={pairs.length <= 1}
                  className="p-2 text-gray-400 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
                  title="Remove pair"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            ))}
          </div>

          {/* Action Buttons */}
          <div className="flex flex-wrap gap-3">
            <button
              type="button"
              onClick={addPair}
              disabled={pairs.length >= 100}
              className="btn-secondary flex items-center gap-2"
            >
              <Plus className="w-4 h-4" />
              Add Pair ({pairs.length}/100)
            </button>

            <button
              type="submit"
              disabled={loading}
              className="btn-primary flex-1 flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  <span>Processing...</span>
                </>
              ) : (
                <>
                  <span>Check All Pairs</span>
                </>
              )}
            </button>

            <button
              type="button"
              onClick={handleClear}
              className="btn-secondary"
            >
              Clear All
            </button>
          </div>
        </form>
      </div>

      {/* Error Display */}
      {error && (
        <div className="mt-6 p-4 bg-red-50 border-2 border-red-200 rounded-lg flex items-start gap-3 animate-slide-up">
          <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
          <div>
            <h4 className="font-semibold text-red-900">Error</h4>
            <p className="text-sm text-red-700">{error}</p>
          </div>
        </div>
      )}

      {/* Results Display */}
      {results && (
        <div className="mt-6 card animate-slide-up">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h3 className="text-xl font-bold text-gray-900">Results</h3>
              <p className="text-sm text-gray-600 mt-1">
                Processed {results.total_pairs} trademark pairs
              </p>
            </div>
            <button
              onClick={exportResults}
              className="btn-secondary flex items-center gap-2"
            >
              <Download className="w-4 h-4" />
              Export CSV
            </button>
          </div>

          {/* Results Table */}
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-200 bg-gray-50">
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase">#</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase">Mark 1</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase">Mark 2</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase">Result</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase">Similarity</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase">Risk</th>
                </tr>
              </thead>
              <tbody>
                {results.results.map((result, index) => (
                  <tr key={index} className="border-b border-gray-100 hover:bg-gray-50 transition-colors">
                    <td className="px-4 py-3 text-sm text-gray-600">#{index + 1}</td>
                    <td className="px-4 py-3 text-sm font-medium text-gray-900">{result.mark1}</td>
                    <td className="px-4 py-3 text-sm font-medium text-gray-900">{result.mark2}</td>
                    <td className="px-4 py-3">
                      <span className={`text-sm font-semibold ${
                        result.label === 1 ? 'text-danger-700' : 'text-success-700'
                      }`}>
                        {result.label_text}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <div className="flex-1 max-w-[100px] bg-gray-200 rounded-full h-2">
                          <div
                            className={`h-full rounded-full ${
                              result.risk_level === 'HIGH'
                                ? 'bg-danger-500'
                                : result.risk_level === 'MEDIUM'
                                ? 'bg-warning-500'
                                : 'bg-success-500'
                            }`}
                            style={{ width: `${result.probability * 100}%` }}
                          />
                        </div>
                        <span className="text-sm font-semibold text-gray-900 w-12">
                          {(result.probability * 100).toFixed(0)}%
                        </span>
                      </div>
                    </td>
                    <td className="px-4 py-3">
                      <span className={`badge ${getRiskBadgeClass(result.risk_level)}`}>
                        {result.risk_level}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Summary Stats */}
          <div className="mt-6 grid grid-cols-3 gap-4 pt-6 border-t border-gray-200">
            <div className="text-center">
              <div className="text-2xl font-bold text-danger-600">
                {results.results.filter(r => r.risk_level === 'HIGH').length}
              </div>
              <div className="text-sm text-gray-600">High Risk</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-warning-600">
                {results.results.filter(r => r.risk_level === 'MEDIUM').length}
              </div>
              <div className="text-sm text-gray-600">Medium Risk</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-success-600">
                {results.results.filter(r => r.risk_level === 'LOW').length}
              </div>
              <div className="text-sm text-gray-600">Low Risk</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default BatchChecker;
