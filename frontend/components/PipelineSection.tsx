"use client";

import React, { useState } from "react";
import { Cpu, FileCode2, Binary, Sliders, BarChart3, ChevronRight } from "lucide-react";

export default function PipelineSection() {
  const [activeStep, setActiveStep] = useState(0);

  const steps = [
    {
      id: "01",
      title: "Defensive Tar & MIME Ingestion",
      icon: FileCode2,
      formula: "Streaming In-Memory Byte Decompression",
      details:
        "Archives are ingested as in-memory streams without extracting unverified members to the local filesystem. Verifies regular files, rejecting directory traversals, symlinks, null bytes, and shell scripts (cmds). Server-generated transport headers and SpamAssassin tags are completely purged to eliminate target leakage.",
      tag: "Phase 02 / Ingestion",
    },
    {
      id: "02",
      title: "Unicode NFKC Normalization",
      icon: Sliders,
      formula: "\\text{NFKC}(\\text{Subject} \\parallel \\text{Body})",
      details:
        "Normalizes multi-byte Unicode variants, strips null bytes, standardizes line breaks, and extracts clean text from HTML structures using a custom parser that removes <script> and <style> without executing arbitrary code. Preserves spam indicators including monetary signs ($) and exclamation points (!).",
      tag: "Phase 04 / Preprocessing",
    },
    {
      id: "03",
      title: "Sublinear TF-IDF Vectorization",
      icon: Binary,
      formula: "\\text{TF-IDF}(t, d) = (1 + \\log \\text{TF}(t, d)) \\times \\log \\frac{1 + N}{1 + \\text{DF}(t)}",
      details:
        "Extracts unigram and bigram tokens with sublinear term-frequency scaling. Capped at 20,000 high-discrimination features. The vectorizer is strictly fitted only on the training partition (70% — 4,145 emails), ensuring zero test partition vocabulary snooping.",
      tag: "Phase 04-05 / Feature Extraction",
    },
    {
      id: "04",
      title: "Multinomial Naïve Bayes",
      icon: Cpu,
      formula: "P(c \\mid \\mathbf{x}) \\propto P(c) \\prod_{i=1}^{n} P(x_i \\mid c)^{x_i}",
      details:
        "Applies Bayes' theorem under feature conditional independence. Lidstone additive smoothing (alpha = 0.1) is mathematically optimized via grid search over validation partitions, solving zero-probability hurdles while preventing over-smoothing on rare tokens.",
      tag: "Phase 05 / Model Tuning",
    },
    {
      id: "05",
      title: "Log-Odds Token Explainability",
      icon: BarChart3,
      formula: "\\text{Contribution}(w) = \\text{TF-IDF}(w) \\times [\\log P(w \\mid \\text{Spam}) - \\log P(w \\mid \\text{Ham})]",
      details:
        "Unlike black-box neural networks, SpamGuard AI decomposes every prediction into exact token-level log-odds contributions. Positive values exert mathematical pull towards Spam, while negative values push towards legitimate Ham.",
      tag: "Phase 07 / Explainability Engine",
    },
  ];

  return (
    <section id="pipeline" className="py-24 px-6 md:px-12 max-w-7xl mx-auto border-t border-white/[0.08]">
      {/* Category Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-6 pb-12 border-b border-white/[0.08]">
        <div>
          <span className="font-mono text-xs uppercase tracking-[0.14em] text-[#A3A3A8] block mb-2">
            02 / The Intelligence
          </span>
          <h2 className="text-4xl sm:text-6xl font-title uppercase tracking-tight text-[#F4F3F0]">
            Mathematical <span className="italic text-[#B8F44A]">Deconstruction</span>
          </h2>
        </div>
        <p className="max-w-md text-sm md:text-base text-[#A3A3A8] font-light leading-relaxed">
          From raw byte stream to probabilistic log-odds calibration. An end-to-end leak-free pipeline engineered strictly without artificial data fabrication.
        </p>
      </div>

      {/* Step Navigation & Interactive Deep Dive */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 my-16">
        {/* Left: Step Selector (Editorial Rows) */}
        <div className="lg:col-span-5 space-y-2">
          {steps.map((s, idx) => {
            const Icon = s.icon;
            const isSelected = activeStep === idx;
            return (
              <button
                key={s.id}
                onClick={() => setActiveStep(idx)}
                className={`w-full text-left p-5 rounded-2xl border transition-all duration-300 flex items-center justify-between ${
                  isSelected
                    ? "bg-[#141418] border-[#B8F44A]/50 shadow-xl shadow-[#B8F44A]/5"
                    : "bg-[#0B0B0D] border-white/[0.06] hover:border-white/[0.15] text-[#A3A3A8] hover:text-[#F4F3F0]"
                }`}
              >
                <div className="flex items-center gap-4">
                  <div
                    className={`w-10 h-10 rounded-xl flex items-center justify-center font-mono text-xs transition-colors ${
                      isSelected
                        ? "bg-[#B8F44A] text-[#0B0B0D] font-bold"
                        : "bg-white/[0.05] text-[#A3A3A8]"
                    }`}
                  >
                    {s.id}
                  </div>
                  <div>
                    <span className="text-[10px] font-mono uppercase tracking-wider text-[#A3A3A8] block">
                      {s.tag}
                    </span>
                    <h3
                      className={`text-base font-semibold transition-colors ${
                        isSelected ? "text-[#F4F3F0]" : "text-[#A3A3A8]"
                      }`}
                    >
                      {s.title}
                    </h3>
                  </div>
                </div>
                <ChevronRight
                  className={`w-4 h-4 transition-transform ${
                    isSelected ? "text-[#B8F44A] translate-x-1" : "text-[#66666E]"
                  }`}
                />
              </button>
            );
          })}
        </div>

        {/* Right: Active Step Mathematical Card */}
        <div className="lg:col-span-7 bg-[#141418] border border-white/[0.08] rounded-3xl p-8 md:p-12 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between pb-6 border-b border-white/[0.08] mb-8">
              <span className="font-mono text-xs text-[#B8F44A] uppercase tracking-widest">
                STEP {steps[activeStep].id} ARCHITECTURE
              </span>
              <span className="font-mono text-xs text-[#A3A3A8]">
                SPAMGUARD CORE PIPELINE
              </span>
            </div>

            <h3 className="text-3xl md:text-4xl font-title text-[#F4F3F0] mb-4">
              {steps[activeStep].title}
            </h3>

            {/* Formula Code Box */}
            <div className="my-6 p-4 rounded-xl bg-[#0B0B0D] border border-white/[0.06] font-mono text-sm text-[#B8F44A] overflow-x-auto">
              <code>{steps[activeStep].formula}</code>
            </div>

            <p className="text-base text-[#A3A3A8] font-light leading-relaxed">
              {steps[activeStep].details}
            </p>
          </div>

          <div className="mt-12 pt-6 border-t border-white/[0.08] flex flex-wrap items-center justify-between text-xs font-mono text-[#A3A3A8] gap-4">
            <span>ISOLATION: 100% UNCONTAMINATED</span>
            <span>VERIFIED ARTIFACT: SPAM_PIPELINE.JOBLIB</span>
          </div>
        </div>
      </div>
    </section>
  );
}
