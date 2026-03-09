import React, { useState } from 'react';
import { Search, Loader2, AlertCircle, Info } from 'lucide-react';
import ResultCard from './ResultCard';
import AnalysisDetails from './AnalysisDetails';
import { API_URL } from '../config';

const SimilarityChecker = () => {
  const [mark1, setMark1] = useState('');
  const [mark2, setMark2] = useState('');
  const [includeDetails, setIncludeDetails] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!mark1.trim() || !mark2.trim()) {
      setError('Please enter both trademark names');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/similarity-check`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          mark1: mark1.trim(),
          mark2: mark2.trim(),
          include_details: includeDetails,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message || 'Failed to check similarity. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setMark1('');
    setMark2('');
    setResult(null);
    setError(null);
  };

  const examplePairs = [
    { mark1: 'SuperCoffee', mark2: 'Super Coffee' },
    { mark1: 'TechSmart', mark2: 'SmartTech' },
    { mark1: 'Nike', mark2: 'Mike' },
  ];

  const loadExample = (example) => {
    setMark1(example.mark1);
    setMark2(example.mark2);
    setResult(null);
    setError(null);
  };

  return (
    <div className="max-w-4xl mx-auto">
      {/* Main Card */}
      <div className="card">
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Input Fields */}
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <label htmlFor="mark1" className="block text-sm font-semibold text-gray-700 mb-2">
                First Trademark
              </label>
              <input
                id="mark1"
                type="text"
                value={mark1}
                onChange={(e) => setMark1(e.target.value)}
                placeholder="e.g., SuperCoffee"
                className="input-field"
                maxLength={200}
              />
              <p className="mt-1 text-xs text-gray-500">{mark1.length}/200 characters</p>
            </div>

            <div>
              <label htmlFor="mark2" className="block text-sm font-semibold text-gray-700 mb-2">
                Second Trademark
              </label>
              <input
                id="mark2"
                type="text"
                value={mark2}
                onChange={(e) => setMark2(e.target.value)}
                placeholder="e.g., Super Coffee"
                className="input-field"
                maxLength={200}
              />
              <p className="mt-1 text-xs text-gray-500">{mark2.length}/200 characters</p>
            </div>
          </div>

          {/* Options */}
          <div className="flex items-center space-x-3 p-4 bg-blue-50 rounded-lg border border-blue-200">
            <input
              id="includeDetails"
              type="checkbox"
              checked={includeDetails}
              onChange={(e) => setIncludeDetails(e.target.checked)}
              className="w-4 h-4 text-primary-600 border-gray-300 rounded focus:ring-primary-500"
            />
            <label htmlFor="includeDetails" className="flex items-center space-x-2 text-sm text-gray-700 cursor-pointer">
              <Info className="w-4 h-4 text-blue-600" />
              <span>Include detailed feature analysis (visual, phonetic, semantic)</span>
            </label>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-4">
            <button
              type="submit"
              disabled={loading || !mark1.trim() || !mark2.trim()}
              className="btn-primary flex-1 flex items-center justify-center space-x-2"
            >
              {loading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  <span>Analyzing...</span>
                </>
              ) : (
                <>
                  <Search className="w-5 h-5" />
                  <span>Check Similarity</span>
                </>
              )}
            </button>

            <button
              type="button"
              onClick={handleClear}
              className="btn-secondary"
            >
              Clear
            </button>
          </div>
        </form>

        {/* Example Pairs */}
        <div className="mt-6 pt-6 border-t border-gray-200">
          <p className="text-sm font-semibold text-gray-700 mb-3">Try these examples:</p>
          <div className="flex flex-wrap gap-2">
            {examplePairs.map((example, index) => (
              <button
                key={index}
                onClick={() => loadExample(example)}
                className="px-4 py-2 text-sm bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg transition-colors border border-gray-300"
              >
                "{example.mark1}" vs "{example.mark2}"
              </button>
            ))}
          </div>
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
      {result && (
        <div className="mt-6 space-y-6 animate-slide-up">
          <ResultCard result={result} mark1={mark1} mark2={mark2} />
          
          {includeDetails && result.details && (
            <AnalysisDetails details={result.details} />
          )}
        </div>
      )}
    </div>
  );
};

export default SimilarityChecker;
