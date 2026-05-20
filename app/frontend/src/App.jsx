import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import RegistrationChecker from './components/RegistrationChecker';
import ApiStatus from './components/ApiStatus';
import { API_URL } from './config';
import { ShieldCheck, Database, Zap } from 'lucide-react';

function App() {
  const [apiStatus, setApiStatus] = useState({ status: 'checking', data: null });

  useEffect(() => {
    checkApiHealth();
  }, []);

  const checkApiHealth = async () => {
    try {
      const response = await fetch(`${API_URL}/health`);
      const data = await response.json();
      setApiStatus({ status: data.status === 'healthy' ? 'healthy' : 'error', data });
    } catch (error) {
      setApiStatus({ status: 'error', data: null });
    }
  };

  return (
    <div className="min-h-screen pb-12">
      <Header />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Hero Section */}
        <div className="text-center mb-12 animate-fade-in">
          <h1 className="text-4xl md:text-5xl font-bold text-gray-900 mb-4">
            Trademark Registration
            <span className="bg-gradient-to-r from-primary-600 to-primary-800 bg-clip-text text-transparent"> Decision System</span>
          </h1>
          <p className="text-lg text-gray-600 max-w-3xl mx-auto mb-6">
            Submit up to 3 trademark names to check against{' '}
            {/* {apiStatus.data?.database_size
              ? <span className="font-semibold text-primary-700">{apiStatus.data.database_size.toLocaleString()}</span>
              : 'thousands of'}{' '} */}
            existing registered trademarks. Powered by Hybrid CNN+SVM with multilingual analysis.
          </p>
          <ApiStatus status={apiStatus} onRetry={checkApiHealth} />
        </div>

        {/* Features Grid */}
        <div className="grid md:grid-cols-3 gap-6 mb-12 animate-slide-up">
          <div className="card text-center">
            <div className="w-12 h-12 bg-primary-100 rounded-lg flex items-center justify-center mx-auto mb-4">
              <ShieldCheck className="w-6 h-6 text-primary-600" />
            </div>
            <h3 className="font-semibold text-gray-900 mb-2">Registration Decision</h3>
            <p className="text-sm text-gray-600">Automatic approve/reject based on similarity threshold</p>
          </div>

          <div className="card text-center">
            <div className="w-12 h-12 bg-primary-100 rounded-lg flex items-center justify-center mx-auto mb-4">
              <Database className="w-6 h-6 text-primary-600" />
            </div>
            <h3 className="font-semibold text-gray-900 mb-2">22K+ Trademark Database</h3>
            <p className="text-sm text-gray-600">Compared against comprehensive existing trademark records</p>
          </div>

          <div className="card text-center">
            <div className="w-12 h-12 bg-primary-100 rounded-lg flex items-center justify-center mx-auto mb-4">
              <Zap className="w-6 h-6 text-primary-600" />
            </div>
            <h3 className="font-semibold text-gray-900 mb-2">Multi-Feature Analysis</h3>
            <p className="text-sm text-gray-600">Visual, phonetic, and semantic similarity detection</p>
          </div>
        </div>

        {/* Main Content */}
        <div className="animate-fade-in">
          <RegistrationChecker />
        </div>
      </div>

      {/* Footer */}
      <footer className="mt-16 py-6 border-t border-gray-200 bg-white/50 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-4 text-center text-gray-600 text-sm">
          <p>Powered by Hybrid CNN+SVM Model &bull; Multilingual Support: English, Hausa, Yoruba</p>
          <p className="mt-2">&copy; 2026 Trademark Registration Decision System. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}

export default App;
