import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import SimilarityChecker from './components/SimilarityChecker';
import BatchChecker from './components/BatchChecker';
import ApiStatus from './components/ApiStatus';
import { API_URL } from './config';
import { CheckCircle2, Layers, Activity } from 'lucide-react';

function App() {
  const [activeTab, setActiveTab] = useState('single'); // 'single' or 'batch'
  const [apiStatus, setApiStatus] = useState({ status: 'checking', data: null });

  // Check API health on mount
  useEffect(() => {
    checkApiHealth();
  }, []);

  const checkApiHealth = async () => {
    try {
      const response = await fetch(`${API_URL}/health`);
      const data = await response.json();
      setApiStatus({ status: 'healthy', data });
    } catch (error) {
      setApiStatus({ status: 'error', data: null });
    }
  };

  return (
    <div className="min-h-screen pb-12">
      <Header />
      
      {/* Hero Section */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="text-center mb-12 animate-fade-in">
          <h1 className="text-4xl md:text-5xl font-bold text-gray-900 mb-4">
            Trademark Similarity
            <span className="bg-gradient-to-r from-primary-600 to-primary-800 bg-clip-text text-transparent"> Engine</span>
          </h1>
          <p className="text-lg text-gray-600 max-w-3xl mx-auto mb-6">
            AI-powered trademark similarity detection using state-of-the-art Hybrid CNN+SVM model 
            with multilingual support for English, Hausa, and Yoruba
          </p>
          
          {/* API Status Badge */}
          <ApiStatus status={apiStatus} onRetry={checkApiHealth} />
        </div>

        {/* Features Grid */}
        <div className="grid md:grid-cols-3 gap-6 mb-12 animate-slide-up">
          <div className="card text-center">
            <div className="w-12 h-12 bg-primary-100 rounded-lg flex items-center justify-center mx-auto mb-4">
              <CheckCircle2 className="w-6 h-6 text-primary-600" />
            </div>
            <h3 className="font-semibold text-gray-900 mb-2">High Accuracy</h3>
            <p className="text-sm text-gray-600">95.4% accuracy with advanced deep learning</p>
          </div>
          
          <div className="card text-center">
            <div className="w-12 h-12 bg-primary-100 rounded-lg flex items-center justify-center mx-auto mb-4">
              <Layers className="w-6 h-6 text-primary-600" />
            </div>
            <h3 className="font-semibold text-gray-900 mb-2">Multilingual</h3>
            <p className="text-sm text-gray-600">Supports English, Hausa, and Yoruba languages</p>
          </div>
          
          <div className="card text-center">
            <div className="w-12 h-12 bg-primary-100 rounded-lg flex items-center justify-center mx-auto mb-4">
              <Activity className="w-6 h-6 text-primary-600" />
            </div>
            <h3 className="font-semibold text-gray-900 mb-2">Real-time Analysis</h3>
            <p className="text-sm text-gray-600">Instant results with detailed feature analysis</p>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex justify-center mb-8">
          <div className="inline-flex rounded-lg border-2 border-gray-200 bg-white p-1 shadow-sm">
            <button
              onClick={() => setActiveTab('single')}
              className={`px-6 py-2.5 rounded-md font-semibold transition-all duration-200 ${
                activeTab === 'single'
                  ? 'bg-primary-600 text-white shadow-md'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              Single Check
            </button>
            <button
              onClick={() => setActiveTab('batch')}
              className={`px-6 py-2.5 rounded-md font-semibold transition-all duration-200 ${
                activeTab === 'batch'
                  ? 'bg-primary-600 text-white shadow-md'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              Batch Check
            </button>
          </div>
        </div>

        {/* Main Content */}
        <div className="animate-fade-in">
          {activeTab === 'single' ? (
            <SimilarityChecker />
          ) : (
            <BatchChecker />
          )}
        </div>
      </div>

      {/* Footer */}
      <footer className="mt-16 py-6 border-t border-gray-200 bg-white/50 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-4 text-center text-gray-600 text-sm">
          <p>Powered by Hybrid CNN+SVM • Model Accuracy: 95.43% • F1 Score: 95.53% • ROC-AUC: 98.76%</p>
          <p className="mt-2">© 2026 Trademark Similarity Engine. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}

export default App;
