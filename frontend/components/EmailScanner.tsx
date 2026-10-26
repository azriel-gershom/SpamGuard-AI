"use client";

import React, { useState } from "react";
import {
  ShieldAlert,
  ShieldCheck,
  Search,
  RotateCcw,
  Sparkles,
  Zap,
  Clock,
  FileText,
  AlertCircle,
  HelpCircle,
} from "lucide-react";
import { classifyEmail, PredictionResponse, FeatureContribution } from "@/lib/api";

export default function EmailScanner() {
  const [subject, setSubject] = useState("");
  const [body, setBody] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Synthetic Demonstration Templates
  const templates = [
    {
      label: "Urgent Phishing Scam (Spam)",
      type: "spam",
      subject: "URGENT: Unauthorized access detected on your account!",
      body:
        "Dear valued customer,\n\n" +
        "We have detected suspicious login attempts on your account from an unrecognized IP address. " +
        "To prevent permanent suspension of your banking access, you must immediately verify your credentials.\n\n" +
        "Click here to verify your identity now: http://secure-verify-update.phishing-test.com/login\n\n" +
        "Failure to verify within 24 hours will result in automatic account termination.\n\n" +
        "Security Team, Global Trust Bank.",
    },
    {
      label: "Promotional Prize Bonus (Spam)",
      type: "spam",
      subject: "Congratulations! You won free cash prize money and bonus offer",
      body:
        "You have been selected as the guaranteed winner of our exclusive $10,000 cash sweepstakes prize! " +
        "Claim your free bonus money today with 100% guarantee payout.\n\n" +
        "Visit our claim form below to receive your immediate wire transfer:\n" +
        "http://claim-free-prize-bonus.com/redeem?id=99283\n\n" +
        "Act fast! This limited bonus offer expires midnight tonight.",
    },
    {
      label: "Engineering Standup (Ham)",
      type: "ham",
      subject: "Sprint 34 Standup Notes & Feature Branch Code Review",
      body:
        "Hi team,\n\n" +
        "Here is the summary of yesterday's sprint retrospective and today's agenda:\n" +
        "1. Sarah merged the authentication pipeline refactoring patch.\n" +
        "2. Dave is addressing unit test edge cases on the reporting module.\n" +
        "3. The staging server deployment is scheduled for 4:00 PM today.\n\n" +
        "Please review the open pull request on git and submit your comments before the sync meeting.\n\n" +
        "Best regards,\nAlex Chen\nLead Software Engineer",
    },
    {
      label: "Academic Submission (Ham)",
      type: "ham",
      subject: "AI Micro Project 05: SpamGuard AI Documentation & Report",
      body:
        "Respected Professor,\n\n" +
        "Please find attached the final project documentation and evaluation metrics for " +
        "Micro Project 05 (AI-Based Spam Email Classifier).\n\n" +
        "We have evaluated the Multinomial Naïve Bayes pipeline against the held-out Apache SpamAssassin " +
        "test partition, achieving 97.30% accuracy, 97.33% spam precision, and 95.51% F1-score.\n\n" +
        "All unit test suites are passing with zero warnings.\n\n" +
        "Sincerely,\nStudent Project Team",
    },
  ];

  const handleSelectTemplate = (tpl: (typeof templates)[0]) => {
    setSubject(tpl.subject);
    setBody(tpl.body);
    setResult(null);
    setError(null);
  };

  const handleClear = () => {
    setSubject("");
    setBody("");
    setResult(null);
    setError(null);
  };

  const handleScan = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!subject.trim() && !body.trim()) {
      setError("Please provide an email subject or body content to classify.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const data = await classifyEmail(subject, body, true, 10);
      setResult(data);
    } catch (err: any) {
      setError(err.message || "Failed to communicate with prediction engine.");
    } finally {
      setLoading(false);
    }
  };

  const totalChars = (subject.length + body.length);
  const spamPct = result ? Math.round(result.spam_probability * 100) : 0;
  const hamPct = result ? Math.round(result.ham_probability * 100) : 0;

  return (
    <section id="scanner" className="py-24 px-6 md:px-12 max-w-7xl mx-auto border-t border-white/[0.08]">
      {/* Category Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-6 pb-12 border-b border-white/[0.08]">
        <div>
          <span className="font-mono text-xs uppercase tracking-[0.14em] text-[#B8F44A] block mb-2">
            03 / Real-Time Classifier
          </span>
          <h2 className="text-4xl sm:text-6xl font-title uppercase tracking-tight text-[#F4F3F0]">
            Single Email <span className="italic text-[#B8F44A]">Scanner</span>
          </h2>
        </div>
        <p className="max-w-md text-sm md:text-base text-[#A3A3A8] font-light leading-relaxed">
          Evaluate any raw email in sub-millisecond real time. View calibrated class probabilities and inspect exact token-level mathematical log-odds contributions.
        </p>
      </div>

      {/* Quick-Fill Demonstration Templates */}
      <div className="mt-8 mb-10">
        <span className="text-xs font-mono uppercase text-[#A3A3A8] tracking-wider block mb-3">
          Quick-Fill Synthetic Test Samples:
        </span>
        <div className="flex flex-wrap gap-2.5">
          {templates.map((tpl) => (
            <button
              key={tpl.label}
              onClick={() => handleSelectTemplate(tpl)}
              className={`px-4 py-2 rounded-full font-mono text-xs transition-all duration-200 border flex items-center gap-2 ${
                tpl.type === "spam"
                  ? "bg-[#FF5D66]/10 text-[#FF5D66] border-[#FF5D66]/20 hover:border-[#FF5D66]/50 hover:bg-[#FF5D66]/20"
                  : "bg-[#54D6A0]/10 text-[#54D6A0] border-[#54D6A0]/20 hover:border-[#54D6A0]/50 hover:bg-[#54D6A0]/20"
              }`}
            >
              <Sparkles className="w-3.5 h-3.5" />
              <span>{tpl.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Main Scanner Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left: Input Form */}
        <div className="lg:col-span-7 bg-[#141418] border border-white/[0.08] rounded-3xl p-6 sm:p-8 flex flex-col justify-between">
          <form onSubmit={handleScan} className="space-y-6">
            <div>
              <label
                htmlFor="subject"
                className="flex items-center justify-between text-xs font-mono uppercase tracking-wider text-[#A3A3A8] mb-2"
              >
                <span>Email Subject Line</span>
                <span>Optional Header</span>
              </label>
              <input
                id="subject"
                type="text"
                value={subject}
                onChange={(e) => setSubject(e.target.value)}
                placeholder="e.g., Claim your $10,000 cash prize..."
                className="w-full px-4 py-3 rounded-xl bg-[#0B0B0D] border border-white/[0.08] focus:border-[#B8F44A] focus:outline-none text-[#F4F3F0] font-sans text-sm transition-colors placeholder:text-[#66666E]"
              />
            </div>

            <div>
              <label
                htmlFor="body"
                className="flex items-center justify-between text-xs font-mono uppercase tracking-wider text-[#A3A3A8] mb-2"
              >
                <span>Email Body Content</span>
                <span>{totalChars.toLocaleString()} / 1,000,000 chars</span>
              </label>
              <textarea
                id="body"
                rows={9}
                value={body}
                onChange={(e) => setBody(e.target.value)}
                placeholder="Paste the full email text, plain message or raw text here..."
                className="w-full p-4 rounded-xl bg-[#0B0B0D] border border-white/[0.08] focus:border-[#B8F44A] focus:outline-none text-[#F4F3F0] font-mono text-xs transition-colors placeholder:text-[#66666E] leading-relaxed resize-y"
              />
            </div>

            {error && (
              <div className="p-4 rounded-xl bg-[#FF5D66]/10 border border-[#FF5D66]/30 text-[#FF5D66] text-xs flex items-center gap-3">
                <AlertCircle className="w-4 h-4 shrink-0" />
                <span>{error}</span>
              </div>
            )}

            {/* Actions */}
            <div className="pt-2 flex flex-wrap items-center justify-between gap-4">
              <button
                type="button"
                onClick={handleClear}
                className="px-5 py-3 rounded-full border border-white/[0.1] text-xs font-mono uppercase text-[#A3A3A8] hover:text-[#F4F3F0] hover:bg-white/[0.05] transition-colors flex items-center gap-2"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                <span>Reset</span>
              </button>

              <button
                type="submit"
                disabled={loading}
                className="px-8 py-3.5 rounded-full bg-[#B8F44A] text-[#0B0B0D] font-mono text-xs uppercase tracking-wider font-semibold hover:bg-[#c9f86f] transition-all shadow-lg shadow-[#B8F44A]/20 flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed hover:scale-[1.02]"
              >
                {loading ? (
                  <>
                    <span className="w-3.5 h-3.5 rounded-full border-2 border-[#0B0B0D] border-t-transparent animate-spin"></span>
                    <span>Analyzing Log-Odds...</span>
                  </>
                ) : (
                  <>
                    <Search className="w-4 h-4" />
                    <span>Analyze Email</span>
                  </>
                )}
              </button>
            </div>
          </form>
        </div>

        {/* Right: Real-Time Results & Explainability */}
        <div className="lg:col-span-5 flex flex-col gap-6">
          {result ? (
            <div
              className={`rounded-3xl p-6 sm:p-8 border transition-all duration-500 flex flex-col justify-between ${
                result.is_spam
                  ? "bg-[#141418] border-[#FF5D66]/40 shadow-2xl shadow-[#FF5D66]/10"
                  : "bg-[#141418] border-[#54D6A0]/40 shadow-2xl shadow-[#54D6A0]/10"
              }`}
            >
              <div>
                {/* Result Banner Header */}
                <div className="flex items-center justify-between pb-4 border-b border-white/[0.08] mb-6 font-mono text-xs">
                  <span className="text-[#A3A3A8] uppercase tracking-wider">
                    CALIBRATED VERDICT
                  </span>
                  <div className="flex items-center gap-1.5 text-[#A3A3A8]">
                    <Clock className="w-3.5 h-3.5 text-[#B8F44A]" />
                    <span>{result.processing_time_ms} ms</span>
                  </div>
                </div>

                {/* Verdict Badge */}
                <div className="flex items-center gap-4 mb-6">
                  <div
                    className={`w-14 h-14 rounded-2xl flex items-center justify-center ${
                      result.is_spam
                        ? "bg-[#FF5D66]/20 text-[#FF5D66] border border-[#FF5D66]/40"
                        : "bg-[#54D6A0]/20 text-[#54D6A0] border border-[#54D6A0]/40"
                    }`}
                  >
                    {result.is_spam ? (
                      <ShieldAlert className="w-8 h-8" />
                    ) : (
                      <ShieldCheck className="w-8 h-8" />
                    )}
                  </div>
                  <div>
                    <span
                      className={`text-2xl sm:text-3xl font-title uppercase tracking-tight block ${
                        result.is_spam ? "text-[#FF5D66]" : "text-[#54D6A0]"
                      }`}
                    >
                      {result.is_spam ? "SPAM DETECTED" : "VERIFIED HAM (SAFE)"}
                    </span>
                    <span className="font-mono text-xs text-[#A3A3A8] uppercase tracking-wider">
                      {result.is_spam
                        ? "Potential Phishing or Deceptive Commercial Vector"
                        : "Legitimate Personal or Work Correspondence"}
                    </span>
                  </div>
                </div>

                {/* Custom Threat Probability Gauge Dial */}
                <div className="my-6 p-5 rounded-2xl bg-[#0B0B0D] border border-white/[0.06]">
                  <div className="flex items-center justify-between font-mono text-xs mb-2">
                    <span className="text-[#A3A3A8] uppercase">Spam Probability</span>
                    <span
                      className={`font-bold ${
                        result.is_spam ? "text-[#FF5D66]" : "text-[#54D6A0]"
                      }`}
                    >
                      {(result.spam_probability * 100).toFixed(2)}%
                    </span>
                  </div>

                  {/* Gradient Dial Bar */}
                  <div className="w-full h-3 bg-white/[0.06] rounded-full overflow-hidden relative">
                    <div
                      className={`h-full rounded-full transition-all duration-700 ${
                        result.is_spam
                          ? "bg-gradient-to-r from-[#B8F44A] via-[#f59e0b] to-[#FF5D66]"
                          : "bg-[#54D6A0]"
                      }`}
                      style={{ width: `${Math.max(spamPct, 3)}%` }}
                    />
                  </div>

                  <div className="flex justify-between font-mono text-[10px] text-[#66666E] uppercase mt-2">
                    <span>Safe (&lt;30%)</span>
                    <span>Suspicious (30-70%)</span>
                    <span>Threat (&gt;70%)</span>
                  </div>
                </div>

                {/* Mathematical Explainability Breakdown */}
                {result.top_features && result.top_features.length > 0 && (
                  <div className="mt-6">
                    <div className="flex items-center justify-between text-xs font-mono uppercase text-[#A3A3A8] mb-3">
                      <span>Token Contribution Breakdown</span>
                      <span className="text-[10px] text-[#66666E]">Log-Odds Diff</span>
                    </div>

                    <div className="space-y-2 max-h-56 overflow-y-auto pr-1">
                      {result.top_features.map((feat, idx) => {
                        const isSpamLeaning = feat.contribution > 0;
                        return (
                          <div
                            key={idx}
                            className="p-2.5 rounded-xl bg-[#0B0B0D] border border-white/[0.04] flex items-center justify-between text-xs font-mono"
                          >
                            <div className="flex items-center gap-2">
                              <span className="text-[#F4F3F0] font-semibold">
                                {feat.token}
                              </span>
                              <span
                                className={`text-[9px] px-1.5 py-0.2 rounded uppercase ${
                                  isSpamLeaning
                                    ? "bg-[#FF5D66]/20 text-[#FF5D66]"
                                    : "bg-[#54D6A0]/20 text-[#54D6A0]"
                                }`}
                              >
                                {isSpamLeaning ? "Spam" : "Ham"}
                              </span>
                            </div>
                            <span
                              className={`font-mono text-xs ${
                                isSpamLeaning ? "text-[#FF5D66]" : "text-[#54D6A0]"
                              }`}
                            >
                              {feat.contribution > 0 ? "+" : ""}
                              {feat.contribution.toFixed(4)}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>

              {/* Bottom Telemetry Strip */}
              <div className="mt-6 pt-4 border-t border-white/[0.06] flex items-center justify-between font-mono text-[10px] text-[#A3A3A8]">
                <span>MODEL: MULTINOMIAL NAÏVE BAYES</span>
                <span>CHARS: {result.cleaned_char_count}</span>
              </div>
            </div>
          ) : (
            /* Standby Card */
            <div className="h-full bg-[#141418] border border-white/[0.08] rounded-3xl p-8 flex flex-col justify-center items-center text-center">
              <div className="w-16 h-16 rounded-full bg-white/[0.03] border border-white/[0.08] flex items-center justify-center text-[#A3A3A8] mb-4">
                <Search className="w-6 h-6 text-[#A3A3A8]" />
              </div>
              <h3 className="text-xl font-title text-[#F4F3F0] mb-2">
                Awaiting Input
              </h3>
              <p className="text-xs text-[#A3A3A8] max-w-xs font-light leading-relaxed mb-6">
                Type or paste an email into the editor or pick one of the synthetic demonstration templates to inspect real-time log-odds inference.
              </p>
              <div className="font-mono text-[10px] text-[#66666E] uppercase border border-white/[0.06] px-3 py-1 rounded-full">
                Zero Client Email Logging Enforced
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
