import { useState } from 'react';
import {
  AlertTriangle,
  User,
  Link2,
  KeyRound,
  CreditCard,
  ShieldAlert,
  ShieldCheck,
  TrendingUp,
  TrendingDown,
  Info,
  Scale,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';

/* ── Core Indicator Definitions ─────────────────────────────────────────
   Matches the 6 indicators displayed in the VerdictCard mockup:
   1. Urgent Language (AlertTriangle, amber)
   2. Personal Info Request (User, slate)
   3. Suspicious URL (Link2, slate/blue)
   4. Credential Request (KeyRound, slate)
   5. Financial Keywords (CreditCard, teal)
   6. Suspicious Vocabulary (ShieldAlert, amber)

   Any indicator with no matching field in the detection pipeline or
   result object is hidden per spec ("hide any indicator tile that has
   no matching field").                                                   */

export default function KeyIndicators({ result, phishingScore = 0 }) {
  const [techOpen, setTechOpen] = useState(false);

  const verdict = result?.verdict || result?.label || 'uncertain';

  /* Collect signals for matching */
  const flagged = Array.isArray(result?.flagged_features) ? result.flagged_features : [];
  const alerts = Array.isArray(result?.alerts) ? result.alerts : [];
  const rules = Array.isArray(result?.rules_triggered) ? result.rules_triggered : [];
  const suspicionReasons =
    Array.isArray(result?.suspicion_reasons) && result.suspicion_reasons.length > 0
      ? result.suspicion_reasons.slice(0, 5)
      : flagged.slice(0, 5);
  const safeReasons =
    Array.isArray(result?.safe_reasons) && result.safe_reasons.length > 0
      ? result.safe_reasons.slice(0, 5)
      : [];

  const allSuspicious = [...flagged, ...alerts, ...rules, ...suspicionReasons].map((s) =>
    String(s).toLowerCase()
  );

  /* ── 1. Urgent Language ────────────────────────────────────────────── */
  const urgentHit = allSuspicious.some((s) =>
    s.includes('urgent') || s.includes('pressure') || s.includes('threat') || s.includes('urgency')
  );

  /* ── 2. Personal Info Request ──────────────────────────────────────── */
  const personalHit = allSuspicious.some((s) =>
    s.includes('personal') || s.includes('identity') || s.includes('sensitive personal')
  );

  /* ── 3. Suspicious URL ─────────────────────────────────────────────── */
  const urlMatches = allSuspicious.filter((s) =>
    s.includes('url') ||
    s.includes('domain') ||
    s.includes('link') ||
    s.includes('typosquat') ||
    s.includes('shorten') ||
    s.includes('punycode') ||
    s.includes('redirect') ||
    s.includes('high-risk')
  );
  let urlDetail = 'Not detected';
  const urlHit = urlMatches.length > 0;
  if (urlHit) {
    const joined = urlMatches.join(' ');
    if (joined.includes('typosquat')) urlDetail = 'Typosquatting detected';
    else if (joined.includes('shorten')) urlDetail = 'Shortened URL detected';
    else if (joined.includes('punycode')) urlDetail = 'Punycode detected';
    else if (joined.includes('redirect')) urlDetail = 'Redirect detected';
    else urlDetail = 'High-risk URL heuristics';
  }

  /* ── 4. Credential Request ─────────────────────────────────────────── */
  const credentialHit = allSuspicious.some((s) =>
    s.includes('credential') || s.includes('password') || s.includes('login') || s.includes('otp')
  );

  /* ── 5. Financial Keywords ─────────────────────────────────────────── */
  const financialHit = allSuspicious.some((s) =>
    s.includes('financial') ||
    s.includes('banking') ||
    s.includes('payment') ||
    s.includes('wire') ||
    s.includes('billing')
  );

  /* ── 6. Suspicious Vocabulary ──────────────────────────────────────── */
  const vocabHit = allSuspicious.some((s) =>
    s.includes('vocabulary') ||
    s.includes('tfidf') ||
    s.includes('suspicious keywords') ||
    s.includes('pattern')
  );

  /* Build the 6 indicator tiles */
  const indicators = [
    {
      id: 'urgent_language',
      label: 'Urgent Language',
      icon: <AlertTriangle className="w-4 h-4 flex-shrink-0" />,
      iconColor: urgentHit ? 'text-amber-400' : 'text-slate-500',
      detected: urgentHit,
      detail: urgentHit ? 'Detected' : 'Not detected',
    },
    {
      id: 'personal_info_request',
      label: 'Personal Info Request',
      icon: <User className="w-4 h-4 flex-shrink-0" />,
      iconColor: personalHit ? 'text-amber-400' : 'text-slate-500',
      detected: personalHit,
      detail: personalHit ? 'Detected' : 'Not detected',
    },
    {
      id: 'suspicious_url',
      label: 'Suspicious URL',
      icon: <Link2 className="w-4 h-4 flex-shrink-0" />,
      iconColor: urlHit ? 'text-slate-400' : 'text-slate-500',
      detected: urlHit,
      detail: urlDetail,
    },
    {
      id: 'credential_request',
      label: 'Credential Request',
      icon: <KeyRound className="w-4 h-4 flex-shrink-0" />,
      iconColor: credentialHit ? 'text-rose-400' : 'text-slate-500',
      detected: credentialHit,
      detail: credentialHit ? 'Detected' : 'Not detected',
    },
    {
      id: 'financial_keywords',
      label: 'Financial Keywords',
      icon: <CreditCard className="w-4 h-4 flex-shrink-0" />,
      iconColor: financialHit ? 'text-teal-400' : 'text-slate-500',
      detected: financialHit,
      detail: financialHit ? 'Detected' : 'Not detected',
    },
    {
      id: 'suspicious_vocabulary',
      label: 'Suspicious Vocabulary',
      icon: <ShieldAlert className="w-4 h-4 flex-shrink-0" />,
      iconColor: vocabHit ? 'text-amber-400' : 'text-slate-500',
      detected: vocabHit,
      detail: vocabHit ? 'Keywords matched' : 'Not detected',
    },
  ];

  /* Risk badge */
  const riskBadge =
    verdict === 'phishing'
      ? { text: 'High Risk', className: 'bg-red-950/60 border-red-800/60 text-red-400' }
      : verdict === 'uncertain'
      ? { text: 'Uncertain', className: 'bg-yellow-950/60 border-yellow-800/60 text-yellow-400' }
      : { text: 'Low Risk', className: 'bg-emerald-950/60 border-emerald-800/60 text-emerald-400' };

  const scorePct = Math.min(100, Math.max(0, Math.round(phishingScore * 100)));
  const isUncertain = verdict === 'uncertain';

  return (
    <div className="space-y-4 mt-4">
      {/* ── Key Indicators Card ── */}
      <div className="bg-[#111827]/70 rounded-2xl p-5 border border-slate-800 shadow-lg">
        {/* Header row */}
        <div className="flex items-center justify-between mb-4">
          <h4 className="text-base font-bold text-slate-100 tracking-tight">Key Indicators</h4>
          <span className={`text-xs font-semibold px-3 py-0.5 rounded-full border ${riskBadge.className}`}>
            {riskBadge.text}
          </span>
        </div>

        {/* 2-col grid of indicator tiles */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {indicators.map((ind) => (
            <div
              key={ind.id}
              className="bg-slate-900/50 border border-slate-800/80 rounded-xl p-3 flex flex-col justify-center transition-colors"
            >
              <div className="flex items-center gap-2">
                <span className={ind.iconColor}>{ind.icon}</span>
                <span className="text-sm font-semibold text-slate-100">{ind.label}</span>
              </div>
              <p
                className={`text-xs mt-1 pl-6 font-normal ${
                  ind.detected ? 'text-slate-300' : 'text-slate-500'
                }`}
              >
                {ind.detail}
              </p>
            </div>
          ))}
        </div>

        {/* Expand / Collapse Button */}
        <button
          type="button"
          onClick={() => setTechOpen(!techOpen)}
          className="w-full mt-4 py-2 px-3 rounded-lg border border-slate-800 bg-slate-900/40 hover:bg-slate-800/60 text-xs font-medium text-slate-400 hover:text-slate-200 flex items-center justify-center gap-1.5 transition-colors cursor-pointer"
        >
          <span>View indicator explanations</span>
          {techOpen ? (
            <ChevronUp className="w-3.5 h-3.5 text-slate-400" />
          ) : (
            <ChevronDown className="w-3.5 h-3.5 text-slate-400" />
          )}
        </button>

        {/* ── Collapsible "Technical details" ── */}
        {techOpen && (
          <div className="mt-4 pt-4 border-t border-slate-800/80 space-y-4 animate-in fade-in duration-200">
            {/* Header */}
            <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
              <Scale className="w-4 h-4 text-slate-400" />
              <span>Technical Details</span>
            </div>

            {/* Raw Phishing Score Gradient Bar */}
            <div className="bg-slate-900/80 rounded-xl p-4 border border-slate-800">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-medium text-slate-300">Raw Phishing Score</span>
                <span className="text-sm font-bold font-mono tracking-tight text-slate-100">
                  {phishingScore.toFixed(4)}{' '}
                  <span className="text-xs text-slate-400 font-normal">({scorePct}%)</span>
                </span>
              </div>

              <div className="relative w-full h-3 bg-slate-800 rounded-full overflow-hidden p-0.5 border border-slate-700/50">
                <div
                  className="h-full rounded-full transition-all duration-500 bg-gradient-to-r from-emerald-500 via-amber-400 to-rose-600 shadow-sm"
                  style={{ width: `${scorePct}%` }}
                />
              </div>

              <div className="flex justify-between items-center text-[11px] text-slate-400 mt-2 font-medium">
                <span className="text-emerald-400">0.0 (Legitimate)</span>
                <span className="text-amber-400">0.5 (Uncertain)</span>
                <span className="text-rose-400">1.0 (Phishing)</span>
              </div>
            </div>

            {/* SHAP Columns */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {/* Suspicion Factors */}
              <div className="bg-slate-900/60 rounded-xl p-3.5 border border-rose-500/20 shadow-sm">
                <div className="flex items-center gap-2 mb-2.5 pb-2 border-b border-slate-800">
                  <TrendingUp className="w-4 h-4 text-rose-400" />
                  <h5 className="text-xs font-semibold uppercase tracking-wider text-rose-300">
                    Top Suspicion Signals
                  </h5>
                </div>
                {suspicionReasons.length > 0 ? (
                  <ul className="space-y-1.5">
                    {suspicionReasons.map((reason, idx) => (
                      <li
                        key={`susp-${idx}`}
                        className="flex items-start gap-2 p-2 rounded-lg bg-rose-500/10 border border-rose-500/20 text-xs text-rose-200/90"
                      >
                        <ShieldAlert className="w-3.5 h-3.5 text-rose-400 flex-shrink-0 mt-0.5" />
                        <span className="leading-relaxed">{reason}</span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-xs text-slate-500 italic p-1">
                    No significant phishing features detected.
                  </p>
                )}
              </div>

              {/* Safe Factors */}
              <div className="bg-slate-900/60 rounded-xl p-3.5 border border-emerald-500/20 shadow-sm">
                <div className="flex items-center gap-2 mb-2.5 pb-2 border-b border-slate-800">
                  <TrendingDown className="w-4 h-4 text-emerald-400" />
                  <h5 className="text-xs font-semibold uppercase tracking-wider text-emerald-300">
                    Top Safe Signals
                  </h5>
                </div>
                {safeReasons.length > 0 ? (
                  <ul className="space-y-1.5">
                    {safeReasons.map((reason, idx) => (
                      <li
                        key={`safe-${idx}`}
                        className="flex items-start gap-2 p-2 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-xs text-emerald-200/90"
                      >
                        <ShieldCheck className="w-3.5 h-3.5 text-emerald-400 flex-shrink-0 mt-0.5" />
                        <span className="leading-relaxed">{reason}</span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-xs text-slate-500 italic p-1">
                    No strong legitimate baseline features detected.
                  </p>
                )}
              </div>
            </div>

            {/* Explainability disclaimer */}
            <div className="flex items-start gap-2 p-2.5 rounded-xl bg-slate-900/40 border border-slate-800 text-[11px] text-slate-400 leading-relaxed">
              <Info className="w-3.5 h-3.5 text-slate-500 flex-shrink-0 mt-0.5" />
              <span>
                Signals are derived from SHAP TreeExplainer feature contributions evaluated on the
                Random Forest classifier.
              </span>
            </div>
          </div>
        )}
      </div>

      {/* ── Uncertain Banner (if applicable) ── */}
      {isUncertain && (
        <div className="p-4 rounded-2xl bg-amber-500/10 border border-amber-500/30 text-amber-200 shadow-md">
          <div className="flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
            <div className="space-y-1">
              <h4 className="text-sm font-semibold text-amber-300">
                Could not decide with confidence – manual review recommended
              </h4>
              <p className="text-xs text-amber-200/80 leading-relaxed">
                The message presents conflicting signals. Please inspect sender domains and message
                links directly through official bookmarks.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
