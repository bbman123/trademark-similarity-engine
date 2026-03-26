import React, { useState } from 'react';
import { CheckCircle2, XCircle, AlertTriangle, ChevronDown, ChevronUp, Eye, Volume2, Globe, Ruler, Languages, ArrowRightLeft } from 'lucide-react';

const RegistrationResult = ({ result, index }) => {
  const [expanded, setExpanded] = useState(false);

  const isApproved = result.decision === 'APPROVED';
  const isRejected = result.decision === 'REJECTED';

  const langColors = {
    'Hausa': 'bg-orange-100 text-orange-800 border-orange-300',
    'Yoruba': 'bg-purple-100 text-purple-800 border-purple-300',
    'English': 'bg-blue-100 text-blue-800 border-blue-300',
    'English (primary)': 'bg-blue-100 text-blue-800 border-blue-300',
  };

  const LanguageBadge = ({ lang }) => {
    if (!lang || lang === 'English') return null;
    const colors = langColors[lang] || 'bg-gray-100 text-gray-700 border-gray-300';
    return (
      <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-semibold border ${colors}`}>
        <Languages className="w-3 h-3" />
        {lang}
      </span>
    );
  };

  const DecisionBadge = () => {
    if (isApproved) {
      return (
        <div className="flex items-center gap-2 px-4 py-2 bg-green-100 text-green-800 rounded-lg">
          <CheckCircle2 className="w-5 h-5" />
          <span className="font-bold text-sm">APPROVED</span>
        </div>
      );
    }
    if (isRejected) {
      return (
        <div className="flex items-center gap-2 px-4 py-2 bg-red-100 text-red-800 rounded-lg">
          <XCircle className="w-5 h-5" />
          <span className="font-bold text-sm">REJECTED</span>
        </div>
      );
    }
    return (
      <div className="flex items-center gap-2 px-4 py-2 bg-yellow-100 text-yellow-800 rounded-lg">
        <AlertTriangle className="w-5 h-5" />
        <span className="font-bold text-sm">{result.decision}</span>
      </div>
    );
  };

  const FeatureSection = ({ title, icon: Icon, children }) => (
    <div className="p-4 bg-gray-50 rounded-lg border border-gray-200">
      <div className="flex items-center gap-2 mb-3">
        <div className="p-1.5 bg-white rounded-lg border border-gray-200">
          <Icon className="w-4 h-4 text-primary-600" />
        </div>
        <h5 className="font-semibold text-gray-900 text-sm">{title}</h5>
      </div>
      <div className="space-y-2">{children}</div>
    </div>
  );

  const FeatureRow = ({ label, value, isBoolean = false }) => (
    <div className="flex items-center justify-between py-1.5 border-b border-gray-200 last:border-0">
      <span className="text-xs text-gray-600">{label}</span>
      {isBoolean ? (
        value ? (
          <span className="flex items-center gap-1 text-xs font-medium text-green-700">
            <CheckCircle2 className="w-3.5 h-3.5" /> Match
          </span>
        ) : (
          <span className="flex items-center gap-1 text-xs font-medium text-gray-500">
            <XCircle className="w-3.5 h-3.5" /> No Match
          </span>
        )
      ) : (
        <span className="text-xs font-semibold text-gray-900">{value}</span>
      )}
    </div>
  );

  const ProgressBar = ({ value, label }) => {
    const percentage = Math.min(Math.max(value * 100, 0), 100);
    return (
      <div className="space-y-1">
        <div className="flex justify-between text-xs text-gray-600">
          <span>{label}</span>
          <span className="font-semibold">{percentage.toFixed(1)}%</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-1.5 overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-primary-500 to-primary-600 transition-all duration-500"
            style={{ width: `${percentage}%` }}
          />
        </div>
      </div>
    );
  };

  const details = result.closest_match?.details;

  return (
    <div className={`card border-2 ${
      isApproved ? 'border-green-200' : isRejected ? 'border-red-200' : 'border-yellow-200'
    }`}>
      {/* Header Row */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1">
          <div className="flex items-center gap-3 mb-2">
            <span className="w-7 h-7 rounded-full bg-gray-100 text-gray-600 flex items-center justify-center text-sm font-bold">
              {index + 1}
            </span>
            <h3 className="text-xl font-bold text-gray-900">"{result.trademark}"</h3>
          </div>
          <p className="text-sm text-gray-600 ml-10">{result.reason}</p>
        </div>
        <DecisionBadge />
      </div>

      {/* Cross-Language Alert */}
      {result.cross_language_note && (
        <div className="mb-4 p-3 bg-amber-50 rounded-lg border border-amber-300">
          <div className="flex items-start gap-2">
            <Languages className="w-4 h-4 text-amber-600 mt-0.5 flex-shrink-0" />
            <div>
              <span className="text-xs font-bold text-amber-800 uppercase tracking-wide">Cross-Language Detection</span>
              <p className="text-sm text-amber-900 mt-1">{result.cross_language_note}</p>
            </div>
          </div>
        </div>
      )}

      {/* Similarity Score Bar */}
      <div className="mb-4 p-3 bg-gray-50 rounded-lg">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-semibold text-gray-700">Max Similarity Score</span>
          <span className={`text-lg font-bold ${
            result.max_similarity >= 0.7 ? 'text-red-600' :
            result.max_similarity >= 0.4 ? 'text-yellow-600' : 'text-green-600'
          }`}>
            {(result.max_similarity * 100).toFixed(1)}%
          </span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
          <div
            className={`h-full transition-all duration-1000 ease-out ${
              result.max_similarity >= 0.7
                ? 'bg-gradient-to-r from-red-500 to-red-600'
                : result.max_similarity >= 0.4
                ? 'bg-gradient-to-r from-yellow-500 to-yellow-600'
                : 'bg-gradient-to-r from-green-500 to-green-600'
            }`}
            style={{ width: `${result.max_similarity * 100}%` }}
          />
        </div>
      </div>

      {/* Closest Match */}
      {result.closest_match && (
        <div className="mb-4 p-3 bg-blue-50 rounded-lg border border-blue-200">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs font-semibold text-blue-700 uppercase tracking-wide">Closest Match</span>
            <LanguageBadge lang={result.closest_match.matched_language} />
          </div>
          <div className="flex items-center justify-between mt-1">
            <div>
              <span className="text-sm font-medium text-gray-900">"{result.closest_match.trademark}"</span>
              {result.closest_match.matched_variant && (
                <span className="text-xs text-blue-600 ml-2">
                  (via {result.closest_match.matched_language}: "{result.closest_match.matched_variant}")
                </span>
              )}
            </div>
            <span className="text-sm font-bold text-blue-700">
              {(result.closest_match.similarity_score * 100).toFixed(1)}% similar
            </span>
          </div>
        </div>
      )}

      {/* Top Matches */}
      {result.top_matches && result.top_matches.length > 1 && (
        <div className="mb-4">
          <button
            onClick={() => setExpanded(!expanded)}
            className="flex items-center gap-2 text-sm font-semibold text-gray-700 hover:text-primary-600 transition-colors"
          >
            {expanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            {expanded ? 'Hide' : 'Show'} top {result.top_matches.length} similar trademarks
          </button>

          {expanded && (
            <div className="mt-3 space-y-2">
              {result.top_matches.map((match, i) => (
                <div key={i} className="p-2 bg-gray-50 rounded border border-gray-200">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2 flex-1 min-w-0">
                      <span className="w-5 h-5 rounded-full bg-gray-200 text-gray-600 flex items-center justify-center text-xs font-bold flex-shrink-0">
                        {i + 1}
                      </span>
                      <span className="text-sm text-gray-800 truncate">{match.trademark}</span>
                      <LanguageBadge lang={match.matched_language} />
                    </div>
                    <span className={`text-sm font-semibold flex-shrink-0 ml-2 ${
                      match.similarity_score >= 0.7 ? 'text-red-600' :
                      match.similarity_score >= 0.4 ? 'text-yellow-600' : 'text-green-600'
                    }`}>
                      {(match.similarity_score * 100).toFixed(1)}%
                    </span>
                  </div>
                  {match.matched_variant && (
                    <div className="ml-7 mt-1 flex items-center gap-1 text-xs text-gray-500">
                      <ArrowRightLeft className="w-3 h-3" />
                      Matched via {match.matched_language} variant: "{match.matched_variant}"
                    </div>
                  )}
                  {/* Translation notes for this match */}
                  {match.details?.translation_notes && match.details.translation_notes.length > 0 && (
                    <div className="ml-7 mt-1 space-y-0.5">
                      {match.details.translation_notes.map((note, ni) => (
                        <div key={ni} className="flex items-center gap-1 text-xs text-amber-700">
                          <Languages className="w-3 h-3 flex-shrink-0" />
                          {note}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ))}

              {/* Detailed Feature Analysis for Closest Match */}
              {details && (
                <div className="mt-4 pt-4 border-t border-gray-200">
                  <h4 className="text-sm font-bold text-gray-900 mb-3">
                    Feature Analysis: "{result.trademark}" vs "{result.closest_match.trademark}"
                  </h4>
                  <div className="grid md:grid-cols-2 gap-4">
                    {details.visual_features && (
                      <FeatureSection title="Visual Similarity" icon={Eye}>
                        <FeatureRow label="Levenshtein Distance" value={details.visual_features.levenshtein_distance} />
                        <ProgressBar value={details.visual_features.jaro_winkler_similarity} label="Jaro-Winkler Similarity" />
                      </FeatureSection>
                    )}
                    {details.phonetic_features && (
                      <FeatureSection title="Phonetic Analysis" icon={Volume2}>
                        <FeatureRow label="Soundex Match" value={details.phonetic_features.soundex_match} isBoolean />
                        <FeatureRow label="Metaphone Match" value={details.phonetic_features.metaphone_match} isBoolean />
                      </FeatureSection>
                    )}
                    {details.semantic_features && (
                      <FeatureSection title="Semantic Similarity" icon={Globe}>
                        <ProgressBar value={details.semantic_features.similarity_en} label="English" />
                        <ProgressBar value={details.semantic_features.similarity_ha} label="Hausa" />
                        <ProgressBar value={details.semantic_features.similarity_yo} label="Yoruba" />
                      </FeatureSection>
                    )}
                    {details.length_features && (
                      <FeatureSection title="Length Analysis" icon={Ruler}>
                        <FeatureRow label="Length Difference" value={`${details.length_features.length_difference} chars`} />
                        <FeatureRow label="Submitted Length" value={`${details.length_features.mark1_length} chars`} />
                        <FeatureRow label="Match Length" value={`${details.length_features.mark2_length} chars`} />
                      </FeatureSection>
                    )}
                    {/* Cross-Language Scoring Details */}
                    {details.cross_language && (
                      <FeatureSection title="Cross-Language Analysis" icon={Languages}>
                        <FeatureRow label="Matched Via" value={details.cross_language.matched_via} />
                        {details.cross_language.matched_variant && (
                          <FeatureRow label="Variant Text" value={`"${details.cross_language.matched_variant}"`} />
                        )}
                        {details.cross_language.variant_scores && (
                          <>
                            {details.cross_language.variant_scores.en !== undefined && (
                              <ProgressBar value={details.cross_language.variant_scores.en} label="English SVM Score" />
                            )}
                            {details.cross_language.variant_scores.ha !== undefined && (
                              <ProgressBar value={details.cross_language.variant_scores.ha} label="Hausa SVM Score" />
                            )}
                            {details.cross_language.variant_scores.yo !== undefined && (
                              <ProgressBar value={details.cross_language.variant_scores.yo} label="Yoruba SVM Score" />
                            )}
                          </>
                        )}
                      </FeatureSection>
                    )}
                    {/* Translation Equivalences */}
                    {details.translation_notes && details.translation_notes.length > 0 && (
                      <FeatureSection title="Translation Equivalences" icon={ArrowRightLeft}>
                        {details.translation_notes.map((note, ni) => (
                          <div key={ni} className="py-1.5 border-b border-gray-200 last:border-0">
                            <span className="text-xs text-gray-800">{note}</span>
                          </div>
                        ))}
                      </FeatureSection>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default RegistrationResult;
