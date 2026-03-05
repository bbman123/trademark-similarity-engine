import React from 'react';
import { Eye, Volume2, Globe, Ruler, CheckCircle2, XCircle } from 'lucide-react';

const AnalysisDetails = ({ details }) => {
  if (!details) return null;

  const FeatureSection = ({ title, icon: Icon, children }) => (
    <div className="p-4 bg-gray-50 rounded-lg border border-gray-200">
      <div className="flex items-center gap-2 mb-3">
        <div className="p-2 bg-white rounded-lg border border-gray-200">
          <Icon className="w-4 h-4 text-primary-600" />
        </div>
        <h4 className="font-semibold text-gray-900">{title}</h4>
      </div>
      <div className="space-y-2">
        {children}
      </div>
    </div>
  );

  const FeatureRow = ({ label, value, isBoolean = false }) => (
    <div className="flex items-center justify-between py-2 border-b border-gray-200 last:border-0">
      <span className="text-sm text-gray-600">{label}</span>
      {isBoolean ? (
        value ? (
          <span className="flex items-center gap-1 text-sm font-medium text-success-700">
            <CheckCircle2 className="w-4 h-4" />
            Match
          </span>
        ) : (
          <span className="flex items-center gap-1 text-sm font-medium text-gray-500">
            <XCircle className="w-4 h-4" />
            No Match
          </span>
        )
      ) : (
        <span className="text-sm font-semibold text-gray-900">{value}</span>
      )}
    </div>
  );

  const ProgressBar = ({ value, max = 1, label }) => {
    const percentage = (value / max) * 100;
    return (
      <div className="space-y-1">
        <div className="flex justify-between text-xs text-gray-600">
          <span>{label}</span>
          <span className="font-semibold">{(value * 100).toFixed(1)}%</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-primary-500 to-primary-600 transition-all duration-500"
            style={{ width: `${percentage}%` }}
          />
        </div>
      </div>
    );
  };

  return (
    <div className="card">
      <div className="flex items-center gap-2 mb-6">
        <h3 className="text-xl font-bold text-gray-900">Detailed Feature Analysis</h3>
        <span className="px-2 py-1 bg-primary-100 text-primary-700 text-xs font-semibold rounded">
          Advanced
        </span>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        {/* Visual Features */}
        {details.visual_features && (
          <FeatureSection title="Visual Similarity" icon={Eye}>
            <FeatureRow
              label="Levenshtein Distance"
              value={details.visual_features.levenshtein_distance}
            />
            <ProgressBar
              value={details.visual_features.jaro_winkler_similarity}
              label="Jaro-Winkler Similarity"
            />
          </FeatureSection>
        )}

        {/* Phonetic Features */}
        {details.phonetic_features && (
          <FeatureSection title="Phonetic Analysis" icon={Volume2}>
            <FeatureRow
              label="Soundex Match"
              value={details.phonetic_features.soundex_match}
              isBoolean
            />
            <FeatureRow
              label="Metaphone Match"
              value={details.phonetic_features.metaphone_match}
              isBoolean
            />
          </FeatureSection>
        )}

        {/* Semantic Features */}
        {details.semantic_features && (
          <FeatureSection title="Semantic Similarity" icon={Globe}>
            <ProgressBar
              value={details.semantic_features.similarity_en}
              label="English"
            />
            <ProgressBar
              value={details.semantic_features.similarity_ha}
              label="Hausa"
            />
            <ProgressBar
              value={details.semantic_features.similarity_yo}
              label="Yoruba"
            />
          </FeatureSection>
        )}

        {/* Length Features */}
        {details.length_features && (
          <FeatureSection title="Length Analysis" icon={Ruler}>
            <FeatureRow
              label="Length Difference"
              value={`${details.length_features.length_difference} characters`}
            />
            <FeatureRow
              label="First Mark Length"
              value={`${details.length_features.mark1_length} chars`}
            />
            <FeatureRow
              label="Second Mark Length"
              value={`${details.length_features.mark2_length} chars`}
            />
          </FeatureSection>
        )}
      </div>

      {/* Model Info */}
      <div className="mt-6 pt-6 border-t border-gray-200">
        <div className="flex flex-wrap gap-4 text-sm text-gray-600">
          {details.cnn_embedding_size && (
            <div className="flex items-center gap-2">
              <span className="font-medium">CNN Embedding:</span>
              <span className="px-2 py-1 bg-blue-100 text-blue-700 rounded font-mono text-xs">
                {details.cnn_embedding_size}D
              </span>
            </div>
          )}
          {details.total_features && (
            <div className="flex items-center gap-2">
              <span className="font-medium">Total Features:</span>
              <span className="px-2 py-1 bg-purple-100 text-purple-700 rounded font-mono text-xs">
                {details.total_features}
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default AnalysisDetails;
