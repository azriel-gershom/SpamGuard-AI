"use client";

import React, { useState } from "react";
import {
  GitCommit,
  Award,
  Users,
  Shield,
  Layers,
  CheckCircle,
  ExternalLink,
  ChevronDown,
  Info,
} from "lucide-react";

export default function Methodology() {
  const [activePhase, setActivePhase] = useState<number | null>(null);

  const phases = [
    {
      num: "01",
      title: "Foundation & Architecture",
      summary: "Directory structuring, virtual environment configuration, smoke test harness.",
      deliverable: "src/config.py, tests/test_smoke.py (4 tests passing)",
    },
    {
      num: "02",
      title: "Dataset Preparation & Sanitization",
      summary: "Defensive in-memory tar extraction, MIME header sanitization, SHA-256 deduplication.",
      deliverable: "src/prepare_data.py, data/raw/emails.csv (11 tests passing)",
    },
    {
      num: "03",
      title: "Exploratory Data Analysis",
      summary: "Statistical distributions, 2.27:1 class imbalance analysis, 6 publication figures.",
      deliverable: "src/analyze_data.py, reports/eda_summary.json (9 tests passing)",
    },
    {
      num: "04",
      title: "Text Preprocessing & Stratified Splitting",
      summary: "Unicode NFKC normalization, leak-free 70/15/15 stratified partition.",
      deliverable: "src/preprocess.py, data/processed/splits/ (14 tests passing)",
    },
    {
      num: "05",
      title: "Naïve Bayes Model Engineering",
      summary: "TF-IDF pipeline assembly, Lidstone alpha=0.1 grid search, Joblib serialization.",
      deliverable: "src/train.py, models/spam_pipeline.joblib (17 tests passing)",
    },
    {
      num: "06",
      title: "Final Performance Evaluation",
      summary: "Frozen evaluation strictly on untouched test partition (n=889), ROC-AUC: 0.9965.",
      deliverable: "src/evaluate.py, reports/test_evaluation.json (16 tests passing)",
    },
    {
      num: "07",
      title: "High-Reliability Prediction Engine",
      summary: "SpamPredictor class, token log-odds explainability, CLI and stdin interface.",
      deliverable: "src/predict.py, CLI entrypoint (21 tests passing)",
    },
    {
      num: "08",
      title: "Streamlit UI Dashboard",
      summary: "Five-tab operations dashboard, Plotly gauge dial, AppTest verification.",
      deliverable: "app.py, src/ui_helpers.py (19 tests passing)",
    },
    {
      num: "09",
      title: "Testing, Security Hardening & Benchmarking",
      summary: "Formula injection defense, DoS bounds, hardware benchmarking (1.04ms latency).",
      deliverable: "reports/performance_benchmark.json (12 tests passing)",
    },
    {
      num: "10",
      title: "Final Academic Submission & Certification",
      summary: "Dataset discrepancy audit resolution, 29-section academic report, viva preparation.",
      deliverable: "docs/SpamGuard_AI_Final_Project_Report.md (Release Certified)",
    },
  ];

  const competencies = [
    {
      code: "CO2",
      title: "Algorithm Selection",
      scope:
        "Justification of Multinomial Naïve Bayes over complex deep architectures for high-dimensional sparse text vectors.",
      evidence: "Phase 05 training benchmarks and Lidstone parameter sensitivity curves.",
    },
    {
      code: "CO4",
      title: "Model Engineering",
      scope:
        "Strict train/test isolation, vocabulary containment, sublinear term-frequency scaling, and defensive feature pipelines.",
      evidence: "Phase 04 leakage audit and zero-overlap content hashing protocols.",
    },
    {
      code: "CO5",
      title: "Evaluation Metrics",
      scope:
        "Multi-metric performance validation exposing the flaw of naive accuracy on imbalanced datasets.",
      evidence: "97.33% Spam Precision, 93.75% Recall, 95.51% F1-score, and 0.9965 ROC-AUC on n=889 test emails.",
    },
  ];

  const teamMembers = [
    {
      name: "Samuvel Raj R",
      role: "Systems Architect & Defensive Engineering",
      focus: "Phases 06–10: High-reliability inference engine, FastAPI integration, security hardening, release certification.",
    },
    {
      name: "Azriel Gershom Raj",
      role: "Machine Learning & Pipeline Engineer",
      focus: "Phases 01–05: Archive ingestion, EDA distributions, text normalization, and Naïve Bayes hyperparameter tuning.",
    },
    {
      name: "CO4 Model Engineering Specialist",
      role: "Feature Extraction & Data Isolation",
      focus: "Stratified partition design, MIME sanitization, unigram/bigram sublinear TF-IDF pipeline.",
    },
    {
      name: "CO5 Quality & Evaluation Specialist",
      role: "Statistical Validation & QA",
      focus: "Held-out test set evaluation, baseline benchmark comparisons, and latency/memory profiling.",
    },
  ];

  return (
    <section id="methodology" className="py-24 px-6 md:px-12 max-w-7xl mx-auto border-t border-white/[0.08]">
      {/* Category Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-6 pb-12 border-b border-white/[0.08]">
        <div>
          <span className="font-mono text-xs uppercase tracking-[0.14em] text-[#B8F44A] block mb-2">
            06 / Research & Methodology
          </span>
          <h2 className="text-4xl sm:text-6xl font-title uppercase tracking-tight text-[#F4F3F0]">
            Academic <span className="italic text-[#B8F44A]">Provenance</span>
          </h2>
        </div>
        <p className="max-w-md text-sm md:text-base text-[#A3A3A8] font-light leading-relaxed">
          Course Competency Mapping, the 10 development phases, verified team roles, and defensive security guarantees.
        </p>
      </div>

      {/* Course Competencies */}
      <div className="my-16">
        <span className="font-mono text-xs uppercase tracking-wider text-[#A3A3A8] block mb-6">
          Integrated Course Competency Mapping:
        </span>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {competencies.map((comp) => (
            <div
              key={comp.code}
              className="p-6 md:p-8 rounded-3xl bg-[#141418] border border-white/[0.08] flex flex-col justify-between"
            >
              <div>
                <span className="font-mono text-xs font-bold text-[#B8F44A] bg-[#B8F44A]/10 px-2.5 py-1 rounded-md border border-[#B8F44A]/20 inline-block mb-4">
                  {comp.code} — {comp.title}
                </span>
                <p className="text-sm text-[#F4F3F0] mb-4 font-light leading-relaxed">
                  {comp.scope}
                </p>
              </div>
              <div className="pt-4 border-t border-white/[0.06] text-xs font-mono text-[#A3A3A8]">
                <span>EVIDENCE: {comp.evidence}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Interactive 10-Phase Timeline */}
      <div className="my-16 bg-[#141418] border border-white/[0.08] rounded-3xl p-8 md:p-12">
        <div className="flex items-center justify-between pb-6 border-b border-white/[0.08] mb-8">
          <div>
            <h3 className="text-2xl font-title text-[#F4F3F0]">
              The 10 Project Development Phases
            </h3>
            <span className="font-mono text-xs text-[#A3A3A8]">
              COMPLETE REPRODUCIBLE LIFECYCLE (PHASE 01 TO 10)
            </span>
          </div>
          <span className="font-mono text-xs text-[#54D6A0] bg-[#54D6A0]/10 px-3 py-1 rounded-full border border-[#54D6A0]/30 hidden sm:inline-block">
            100% COMPLETE & VERIFIED
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {phases.map((p, idx) => (
            <div
              key={p.num}
              onClick={() => setActivePhase(activePhase === idx ? null : idx)}
              className="p-5 rounded-2xl bg-[#0B0B0D] border border-white/[0.06] hover:border-white/[0.18] transition-all cursor-pointer flex flex-col justify-between"
            >
              <div className="flex items-center justify-between mb-2">
                <span className="font-mono text-xs text-[#B8F44A] font-bold">
                  PHASE {p.num}
                </span>
                <CheckCircle className="w-4 h-4 text-[#54D6A0]" />
              </div>
              <h4 className="text-base font-semibold text-[#F4F3F0] mb-1">
                {p.title}
              </h4>
              <p className="text-xs text-[#A3A3A8] font-light leading-relaxed mb-3">
                {p.summary}
              </p>
              <div className="pt-2 border-t border-white/[0.04] text-[10px] font-mono text-[#66666E]">
                {p.deliverable}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Team Roster Grid */}
      <div className="my-16">
        <span className="font-mono text-xs uppercase tracking-wider text-[#A3A3A8] block mb-6">
          Verified Academic Research Team:
        </span>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {teamMembers.map((member, idx) => (
            <div
              key={idx}
              className="p-6 rounded-2xl bg-[#141418] border border-white/[0.08] flex flex-col justify-between"
            >
              <div>
                <div className="w-10 h-10 rounded-full bg-white/[0.04] border border-white/[0.08] flex items-center justify-center text-[#B8F44A] mb-4 font-mono font-bold text-xs">
                  0{idx + 1}
                </div>
                <h4 className="text-base font-semibold text-[#F4F3F0] mb-1">
                  {member.name}
                </h4>
                <span className="font-mono text-xs text-[#B8F44A] block mb-3">
                  {member.role}
                </span>
                <p className="text-xs text-[#A3A3A8] font-light leading-relaxed">
                  {member.focus}
                </p>
              </div>
              <div className="mt-4 pt-3 border-t border-white/[0.04] font-mono text-[10px] text-[#66666E]">
                VERIFIED COLLABORATOR
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Security & Limitations Disclosures */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 my-16">
        {/* Security Controls */}
        <div className="p-8 rounded-3xl bg-[#141418] border border-white/[0.08]">
          <div className="flex items-center gap-3 mb-4">
            <Shield className="w-5 h-5 text-[#B8F44A]" />
            <h4 className="text-xl font-title text-[#F4F3F0]">
              Defensive Hardening Controls
            </h4>
          </div>
          <ul className="space-y-3 font-mono text-xs text-[#A3A3A8]">
            <li className="flex items-start gap-2">
              <span className="text-[#B8F44A]">✓</span>
              <span>Formula Injection Defense: Prefixes (=, +, -, @) automatically sanitized.</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-[#B8F44A]">✓</span>
              <span>DoS Payload Bounds: Capped at 1,000,000 characters and 10,000 batch rows.</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-[#B8F44A]">✓</span>
              <span>Zero User Data Logging: No incoming emails are persisted to disk or external cloud.</span>
            </li>
          </ul>
        </div>

        {/* Limitations & Provenance */}
        <div className="p-8 rounded-3xl bg-[#141418] border border-white/[0.08]">
          <div className="flex items-center gap-3 mb-4">
            <Info className="w-5 h-5 text-[#f59e0b]" />
            <h4 className="text-xl font-title text-[#F4F3F0]">
              Academic Scope & Limitations
            </h4>
          </div>
          <p className="text-xs text-[#A3A3A8] leading-relaxed font-light mb-4">
            SpamGuard AI was trained and evaluated on the authentic Apache SpamAssassin Public Corpus (5,923 emails). While achieving 97.30% accuracy and 0.9965 ROC-AUC, vocabulary distributions reflect early-2000s developer mailing lists. Modern zero-day threats require ongoing corpus refresh cycles.
          </p>
          <div className="font-mono text-[10px] text-[#A3A3A8]">
            TRANSPARENT DISCLOSURE AS PER PHASE 10 SUBMISSION GATE
          </div>
        </div>
      </div>
    </section>
  );
}
