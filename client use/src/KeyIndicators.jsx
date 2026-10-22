import { useState } from 'react';
import {
  AlertTriangle,
  User,
  Link as LinkIcon,
  Key,
  CreditCard,
  Gift,
  MessageSquare,
  ShieldAlert,
  ChevronDown,
  ChevronUp,
  Info,
  AlertCircle
} from 'lucide-react';

export default function KeyIndicators({ result, phishingScore = 0 }) {
  const [isExpanded, setIsExpanded] = useState(false);

  const flagged = Array.isArray(result?.flagged_features) ? result.flagged_features : [];

  // ── 1. Urgent Language ──────────────────────────────────────────────
  const urgentMatch = flagged.find(f =>
    f.toLowerCase().includes("threat and urgency") ||
    f.toLowerCase().includes("urgent or pressure") ||
    f.toLowerCase().includes("threatening or consequence") ||
    f.toLowerCase().includes("urgency")
  );
  const isUrgent = Boolean(urgentMatch);
  let urgentSubtitle = "Not detected";
  let urgentWhat = "No high-pressure keywords, threatening consequences, or artificial deadlines detected.";
  if (isUrgent) {
    urgentSubtitle = "Keywords found";
    urgentWhat = urgentMatch || "High-pressure urgency and consequence keywords detected.";
  }

  // ── 2. Personal Info Request ────────────────────────────────────────
  const personalInfoMatch = flagged.find(f =>
    f.toLowerCase().includes("personal info") ||
    f.toLowerCase().includes("direct request for sensitive credentials or personal info")
  );
  const isPersonalInfo = Boolean(personalInfoMatch);
  let personalSubtitle = "Not detected";
  let personalWhat = "No direct requests for sensitive personal identity information or documents.";
  if (isPersonalInfo) {
    personalSubtitle = "Keywords found";
    personalWhat = personalInfoMatch || "Direct request for sensitive personal identity information identified.";
  }

  // ── 3. Suspicious URL ───────────────────────────────────────────────
  const urlMatches = flagged.filter(f =>
    f.toLowerCase().includes("url") ||
    f.toLowerCase().includes("domain") ||
    f.toLowerCase().includes("typosquatting") ||
    f.toLowerCase().includes("shortener") ||
    f.toLowerCase().includes("subdomain") ||
    f.toLowerCase().includes("raw ip") ||
    f.toLowerCase().includes("special character") ||
    f.toLowerCase().includes("punycode") ||
    f.toLowerCase().includes("redirect") ||
    f.toLowerCase().includes("blacklisted") ||
    f.toLowerCase().includes("mismatch")
  );
  const isSuspiciousUrl = urlMatches.length > 0;
  let urlSubtitle = "Not detected";
  let urlWhat = "No suspicious links, deceptive domains, or abnormal URL structures detected.";
  if (isSuspiciousUrl) {
    const firstMatch = urlMatches[0].toLowerCase();
    if (firstMatch.includes("shortener")) {
      urlSubtitle = "Shortened URL detected";
    } else if (firstMatch.includes("typosquatting")) {
      urlSubtitle = "Typosquatting risk detected";
    } else if (firstMatch.includes("blacklisted")) {
      urlSubtitle = "Blacklisted domain detected";
    } else if (firstMatch.includes("raw ip")) {
      urlSubtitle = "Raw IP address link";
    } else if (firstMatch.includes("mismatch") || firstMatch.includes("does not match")) {
      urlSubtitle = "Brand domain mismatch";
    } else if (firstMatch.includes("subdomain")) {
      urlSubtitle = "Excessive subdomains";
    } else if (firstMatch.includes("punycode")) {
      urlSubtitle = "Obfuscated Punycode URL";
    } else if (firstMatch.includes("redirect")) {
      urlSubtitle = "Cross-domain redirect";
    } else if (firstMatch.includes("deceptive")) {
      urlSubtitle = "Deceptive link keywords";
    } else if (firstMatch.includes("special character")) {
      urlSubtitle = "High special characters";
    } else if (firstMatch.includes("heuristics triggered")) {
      urlSubtitle = "High-risk URL heuristics";
    } else {
      urlSubtitle = "Suspicious link detected";
    }
    urlWhat = urlMatches.join(". ");
  }

  // ── 4. Credential Request ───────────────────────────────────────────
  const credentialMatch = flagged.find(f =>
    f.toLowerCase().includes("credentials") ||
    f.toLowerCase().includes("login/payment") ||
    f.toLowerCase().includes("password")
  );
  const isCredential = Boolean(credentialMatch);
  let credentialSubtitle = "Not detected";
  let credentialWhat = "No prompts requesting passwords, OTPs, PINs, or login credentials.";
  if (isCredential) {
    credentialSubtitle = "Keywords found";
    credentialWhat = credentialMatch || "Prompts soliciting login credentials or authentication details detected.";
  }

  // ── 5. Financial Keywords ───────────────────────────────────────────
  const financialMatch = flagged.find(f =>
    f.toLowerCase().includes("financial") ||
    f.toLowerCase().includes("banking")
  );
  const isFinancial = Boolean(financialMatch);
  let financialSubtitle = "Not detected";
  let financialWhat = "No banking, payment transaction, or wire transfer terminology found.";
  if (isFinancial) {
    financialSubtitle = "Keywords found";
    financialWhat = financialMatch || "Financial or banking context detected in message text.";
  }

  // ── Core Evaluated Indicators ───────────────────────────────────────
  const coreIndicators = [
    {
      id: 'urgent_language',
      label: 'Urgent Language',
      icon: <AlertTriangle className="w-4 h-4 flex-shrink-0" />,
      detected: isUrgent,
      subtitle: urgentSubtitle,
      statusBadge: isUrgent ? 'Keywords found' : 'Not detected',
      color: 'amber',
      whatWasDetected: urgentWhat,
      whySuspicious: 'Social engineering attacks frequently create artificial panic or urgent deadlines to induce impulsive compliance before the recipient can verify authenticity.',
      riskImpact: 'Heavily weighted in phishing classification; urgency combined with call-to-actions is a primary hallmark of phishing campaigns.'
    },
    {
      id: 'personal_info_request',
      label: 'Personal Info Request',
      icon: <User className="w-4 h-4 flex-shrink-0" />,
      detected: isPersonalInfo,
      subtitle: personalSubtitle,
      statusBadge: isPersonalInfo ? 'Keywords found' : 'Not detected',
      color: 'amber',
      whatWasDetected: personalWhat,
      whySuspicious: 'Solicitations for sensitive identification details (such as national ID numbers, Aadhaar, PAN, SSN, or identity documents) are standard vectors for identity theft.',
      riskImpact: 'Significantly increases the risk score; trusted organizations rarely solicit sensitive personal IDs through unauthenticated messages.'
    },
    {
      id: 'suspicious_url',
      label: 'Suspicious URL',
      icon: <LinkIcon className="w-4 h-4 flex-shrink-0" />,
      detected: isSuspiciousUrl,
      subtitle: urlSubtitle,
      statusBadge: isSuspiciousUrl ? 'Detected' : 'Not detected',
      color: 'blue',
      whatWasDetected: urlWhat,
      whySuspicious: 'Deceptive web links (typosquatted domains, URL shorteners, raw IP links, or brand mismatches) are the primary vector used to route users to fake credential-harvesting portals.',
      riskImpact: 'Critical risk factor; high-risk URL heuristics or blacklist matches strongly push predictions toward phishing.'
    },
    {
      id: 'credential_request',
      label: 'Credential Request',
      icon: <Key className="w-4 h-4 flex-shrink-0" />,
      detected: isCredential,
      subtitle: credentialSubtitle,
      statusBadge: isCredential ? 'Keywords found' : 'Not detected',
      color: 'rose',
      whatWasDetected: credentialWhat,
      whySuspicious: 'Directly asking for passwords, one-time passwords (OTPs), or secret security codes is an immediate red flag for credential theft and account takeover.',
      riskImpact: 'Strong indicator of credential harvesting; elevates risk when paired with links or urgent prompts.'
    },
    {
      id: 'financial_keywords',
      label: 'Financial Keywords',
      icon: <CreditCard className="w-4 h-4 flex-shrink-0" />,
      detected: isFinancial,
      subtitle: financialSubtitle,
      statusBadge: isFinancial ? 'Keywords found' : 'Not detected',
      color: 'emerald',
      whatWasDetected: financialWhat,
      whySuspicious: 'Scammers frequently fabricate banking notices, unauthorized transaction alerts, or overdue invoice penalties to trigger high emotional concern.',
      riskImpact: 'Acts as an aggravating contextual multiplier when combined with urgency or credential requests.'
    }
  ];

  // ── Optional Additional Pipeline Indicators (only if flagged) ───────
  const additionalIndicators = [];

  const rewardMatch = flagged.find(f => f.toLowerCase().includes("prize") || f.toLowerCase().includes("reward") || f.toLowerCase().includes("bait"));
  if (rewardMatch) {
    additionalIndicators.push({
      id: 'reward_bait',
      label: 'Reward / Bait Language',
      icon: <Gift className="w-4 h-4 flex-shrink-0" />,
      detected: true,
      subtitle: 'Detected',
      color: 'purple',
      whatWasDetected: rewardMatch,
      whySuspicious: 'Offers of unexpected lottery winnings, prizes, or gift cards are used to lure victims into interacting with malicious links.',
      riskImpact: 'Increases deception and social engineering probability in the model.'
    });
  }

  const greetingMatch = flagged.find(f => f.toLowerCase().includes("impersonal") || f.toLowerCase().includes("generic greeting"));
  if (greetingMatch) {
    additionalIndicators.push({
      id: 'generic_greeting',
      label: 'Generic Greeting',
      icon: <MessageSquare className="w-4 h-4 flex-shrink-0" />,
      detected: true,
      subtitle: 'Detected',
      color: 'indigo',
      whatWasDetected: greetingMatch,
      whySuspicious: 'Mass phishing operations usually do not know the recipient’s real name and rely on generic salutations like "Dear Customer" or "Dear User".',
      riskImpact: 'Contributes mild spam/phishing weighting in feature analysis.'
    });
  }

  const tfidfMatch = flagged.find(f => f.toLowerCase().includes("suspicious vocabulary"));
  if (tfidfMatch) {
    additionalIndicators.push({
      id: 'suspicious_vocabulary',
      label: 'Suspicious Vocabulary',
      icon: <ShieldAlert className="w-4 h-4 flex-shrink-0" />,
      detected: true,
      subtitle: 'Keywords matched',
      color: 'amber',
      whatWasDetected: tfidfMatch,
      whySuspicious: 'Specific keyword co-occurrences align closely with patterns learned from historical phishing corpora.',
      riskImpact: 'Raises the TF-IDF feature score in the Random Forest vectorizer.'
    });
  }

  const templateMatch = flagged.find(f => f.toLowerCase().includes("known phishing templates"));
  if (templateMatch) {
    additionalIndicators.push({
      id: 'template_anomaly',
      label: 'Template Pattern',
      icon: <AlertCircle className="w-4 h-4 flex-shrink-0" />,
      detected: true,
      subtitle: 'Template matched',
      color: 'orange',
      whatWasDetected: templateMatch,
      whySuspicious: 'Linguistic syntax and sentence structure mirror documented phishing campaign templates.',
      riskImpact: 'Reflects high structural similarity evaluated by the machine learning model.'
    });
  }

  const allIndicators = [...coreIndicators, ...additionalIndicators];

  // ── Risk Badge Styling ──────────────────────────────────────────────
  const getRiskBadge = () => {
    if (phishingScore >= 0.7) {
      return {
        label: 'High Risk',
        className: 'bg-red-500/15 text-red-400 border-red-500/30'
      };
    } else if (phishingScore >= 0.35) {
      return {
        label: 'Moderate Risk',
        className: 'bg-amber-500/15 text-amber-400 border-amber-500/30'
      };
    } else {
      return {
        label: 'Low Risk',
        className: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30'
      };
    }
  };

  const riskBadge = getRiskBadge();

  return (
    <div className="bg-slate-900/60 rounded-2xl p-5 border border-slate-700/60 shadow-lg mt-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h4 className="text-base font-semibold text-slate-100 tracking-tight">
          Key Indicators
        </h4>
        <span
          className={`px-3 py-1 rounded-full text-xs font-semibold border ${riskBadge.className} transition-colors`}
        >
          {riskBadge.label}
        </span>
      </div>

      {/* Compact Indicators Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {allIndicators.map((ind) => {
          const isDetected = ind.detected;
          return (
            <div
              key={ind.id}
              className={`p-3.5 rounded-xl border transition-all duration-200 flex flex-col justify-between ${
                isDetected
                  ? 'bg-slate-900/80 border-slate-700/80 hover:border-slate-600 shadow-sm'
                  : 'bg-slate-900/30 border-slate-800/80 hover:border-slate-800/90'
              }`}
            >
              <div className="flex items-center gap-2.5">
                <span
                  className={
                    isDetected
                      ? ind.color === 'blue'
                        ? 'text-blue-400'
                        : ind.color === 'rose'
                        ? 'text-rose-400'
                        : ind.color === 'emerald'
                        ? 'text-emerald-400'
                        : ind.color === 'purple'
                        ? 'text-purple-400'
                        : 'text-amber-400'
                      : 'text-slate-500'
                  }
                >
                  {ind.icon}
                </span>
                <span
                  className={`text-sm font-medium leading-tight ${
                    isDetected ? 'text-slate-100' : 'text-slate-300'
                  }`}
                >
                  {ind.label}
                </span>
              </div>
              <p
                className={`text-xs mt-1.5 pl-6.5 break-words leading-relaxed ${
                  isDetected
                    ? 'text-slate-300 font-medium'
                    : 'text-slate-500 font-normal'
                }`}
              >
                {ind.subtitle}
              </p>
            </div>
          );
        })}
      </div>

      {/* Expandable "View Details" Section */}
      {isExpanded && (
        <div className="mt-4 pt-4 border-t border-slate-700/50 space-y-3">
          {allIndicators.map((ind) => (
            <div
              key={`detail-${ind.id}`}
              className={`p-3.5 rounded-xl border text-left transition-all ${
                ind.detected
                  ? 'bg-slate-950/60 border-slate-700/60'
                  : 'bg-slate-950/30 border-slate-800/60'
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <span
                    className={
                      ind.detected
                        ? ind.color === 'blue'
                          ? 'text-blue-400'
                          : ind.color === 'rose'
                          ? 'text-rose-400'
                          : ind.color === 'emerald'
                          ? 'text-emerald-400'
                          : 'text-amber-400'
                        : 'text-slate-500'
                    }
                  >
                    {ind.icon}
                  </span>
                  <span className="text-sm font-semibold text-slate-200">
                    {ind.label}
                  </span>
                </div>
                <span
                  className={`text-[11px] font-medium px-2 py-0.5 rounded-md border ${
                    ind.detected
                      ? 'bg-amber-500/10 text-amber-300 border-amber-500/30'
                      : 'bg-slate-800/80 text-slate-400 border-slate-700/50'
                  }`}
                >
                  {ind.statusBadge || (ind.detected ? 'Detected' : 'Not detected')}
                </span>
              </div>

              <div className="space-y-2 mt-2 text-xs">
                <div>
                  <span className="font-semibold text-slate-300">What was evaluated: </span>
                  <span className="text-slate-400 leading-relaxed">{ind.whatWasDetected}</span>
                </div>
                <div>
                  <span className="font-semibold text-slate-300">Why it may be suspicious: </span>
                  <span className="text-slate-400 leading-relaxed">{ind.whySuspicious}</span>
                </div>
                <div>
                  <span className="font-semibold text-slate-300">Risk assessment contribution: </span>
                  <span className="text-slate-400 leading-relaxed">{ind.riskImpact}</span>
                </div>
              </div>
            </div>
          ))}

          {/* Educational Disclaimer */}
          <div className="flex items-start gap-2.5 p-3 rounded-xl bg-blue-500/10 border border-blue-500/20 text-xs text-blue-200/90 leading-relaxed mt-2">
            <Info className="w-4 h-4 text-blue-400 flex-shrink-0 mt-0.5" />
            <div>
              <strong className="text-blue-300 font-semibold">Evaluation Note: </strong>
              Indicators represent behavioral and structural signals evaluated collectively by the detection model. An individual indicator is not definitive proof of phishing on its own.
            </div>
          </div>
        </div>
      )}

      {/* Expand / Collapse Button */}
      <button
        type="button"
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full mt-4 py-2.5 px-4 rounded-xl border border-slate-700/60 bg-slate-800/40 hover:bg-slate-800/70 text-slate-300 text-xs font-medium flex items-center justify-center gap-2 transition-all duration-200 hover:border-slate-600 focus:outline-none focus:ring-2 focus:ring-blue-500/40 cursor-pointer"
        aria-expanded={isExpanded}
      >
        <span>{isExpanded ? 'Hide indicator explanations' : 'View indicator explanations'}</span>
        {isExpanded ? (
          <ChevronUp className="w-4 h-4 text-slate-400 transition-transform" />
        ) : (
          <ChevronDown className="w-4 h-4 text-slate-400 transition-transform" />
        )}
      </button>
    </div>
  );
}
