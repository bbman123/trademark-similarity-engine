import React from 'react';
import { AlertTriangle, CheckCircle2, XCircle, TrendingUp, Shield } from 'lucide-react';

const ResultCard = ({ result, mark1, mark2 }) => {
  const getRiskIcon = (risk) => {
    switch (risk) {
      case 'HIGH':
        return <AlertTriangle className="w-6 h-6" />;
      case 'MEDIUM':
        return <AlertTriangle className="w-6 h-6" />;
      case 'LOW':
        return <CheckCircle2 className="w-6 h-6" />;
      default:
        return <XCircle className="w-6 h-6" />;
    }
  };

  const getRiskColor = (risk) => {
    switch (risk) {
      case 'HIGH':
        return 'danger';
      case 'MEDIUM':
        return 'warning';
      case 'LOW':
        return 'success';
      default:
        return 'gray';
    }
  };

  const riskColor = getRiskColor(result.risk_level);
  const probability = (result.probability * 100).toFixed(1);

  return (
    <div className="card">
      {/* Header */}
      <div className="flex items-start justify-between mb-6">
        <div className="flex-1">
          <h3 className="text-2xl font-bold text-gray-900 mb-2">Similarity Analysis</h3>
          <div className="flex items-center gap-3 text-sm text-gray-600">
            <span className="font-medium">"{mark1}"</span>
            <span className="text-gray-400">vs</span>
            <span className="font-medium">"{mark2}"</span>
          </div>
        </div>
        <div className={`p-3 rounded-full bg-${riskColor}-100`}>
          {getRiskIcon(result.risk_level)}
        </div>
      </div>

      {/* Main Result */}
      <div className="grid md:grid-cols-2 gap-6 mb-6">
        {/* Similarity Score */}
        <div className="relative">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-semibold text-gray-700">Similarity Score</span>
            <span className="text-2xl font-bold text-gray-900">{probability}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
            <div
              className={`h-full bg-gradient-to-r transition-all duration-1000 ease-out ${
                result.risk_level === 'HIGH'
                  ? 'from-danger-500 to-danger-600'
                  : result.risk_level === 'MEDIUM'
                  ? 'from-warning-500 to-warning-600'
                  : 'from-success-500 to-success-600'
              }`}
              style={{ width: `${probability}%` }}
            />
          </div>
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>0%</span>
            <span>50%</span>
            <span>100%</span>
          </div>
        </div>

        {/* Risk Level */}
        <div className="flex flex-col justify-center">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-semibold text-gray-700">Risk Assessment</span>
            <span className={`badge badge-${result.risk_level.toLowerCase()}`}>
              {result.risk_level} RISK
            </span>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <Shield className="w-4 h-4 text-gray-500" />
            <span className="text-gray-600">
              {result.label_text === 'Similar' ? 'Trademarks are similar' : 'Trademarks are dissimilar'}
            </span>
          </div>
        </div>
      </div>

      {/* Recommendation */}
      <div className={`p-4 bg-${riskColor}-50 border-2 border-${riskColor}-200 rounded-lg`}>
        <div className="flex items-start gap-3">
          <TrendingUp className={`w-5 h-5 text-${riskColor}-600 flex-shrink-0 mt-0.5`} />
          <div>
            <h4 className={`font-semibold text-${riskColor}-900 mb-1`}>Recommendation</h4>
            <p className={`text-sm text-${riskColor}-700`}>{result.recommendation}</p>
          </div>
        </div>
      </div>

      {/* Confidence Meter */}
      <div className="mt-6 pt-6 border-t border-gray-200">
        <div className="flex items-center justify-between text-sm">
          <span className="text-gray-600">Model Confidence</span>
          <span className="font-semibold text-gray-900">{(result.confidence * 100).toFixed(1)}%</span>
        </div>
      </div>
    </div>
  );
};

export default ResultCard;
