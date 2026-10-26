"use client";

import React, { useState } from "react";
import Image from "next/image";
import {
  BarChart2,
  TrendingUp,
  ShieldCheck,
  AlertOctagon,
  Layers,
  FileText,
  Eye,
  Maximize2,
  CheckCircle,
} from "lucide-react";

export default function ModelAnalytics() {
  const [selectedFigure, setSelectedFigure] = useState<string | null>(null);

  const figures = [
    {
      title: "Confusion Matrix Heatmap",
      filename: "test_confusion_matrix.png",
      caption: "Held-out test set (n=889). 610 TN, 7 FP (1.13%), 17 FN (6.25%), 255 TP.",
    },
    {
      title: "Receiver Operating Characteristic (ROC)",
      filename: "test_roc_curve.png",
      caption: "ROC-AUC: 0.9965. Demonstrating near-perfect class separability across decision thresholds.",
    },
    {
      title: "Precision-Recall Curve",
      filename: "test_precision_recall_curve.png",
      caption: "Average Precision: 0.9926. Verifying high precision even under heavy recall demands.",
    },
    {
      title: "Model vs Majority Baseline",
      filename: "test_model_vs_baseline.png",
      caption: "MNB (97.3% Acc, 95.51% F1) vs Majority Baseline (69.4% Acc, 0% F1).",
    },
    {
      title: "Top Discriminative Spam Vocabulary",
      filename: "top_spam_words.png",
      caption: "Top non-stopword tokens in spam: font, email, free, click, money, business.",
    },
    {
      title: "Top Discriminative Ham Vocabulary",
      filename: "top_ham_words.png",
      caption: "Top non-stopword tokens in ham: list, linux, users, org, rpm, message.",
    },
  ];

  return (
    <section id="analytics" className="py-24 px-6 md:px-12 max-w-7xl mx-auto border-t border-white/[0.08]">
      {/* Category Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-6 pb-12 border-b border-white/[0.08]">
        <div>
          <span className="font-mono text-xs uppercase tracking-[0.14em] text-[#B8F44A] block mb-2">
            05 / Empirical Certification
          </span>
          <h2 className="text-4xl sm:text-6xl font-title uppercase tracking-tight text-[#F4F3F0]">
            Model <span className="italic text-[#B8F44A]">Intelligence</span>
          </h2>
        </div>
        <div className="flex flex-col items-start md:items-end font-mono text-xs text-[#A3A3A8]">
          <span>EVALUATION SPLIT: TEST.CSV (n=889 EMAILS)</span>
          <span className="text-[#B8F44A]">ZERO FIT RETRAINING AT INFERENCE</span>
        </div>
      </div>

      {/* Giant Performance Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-6 my-16 font-mono">
        <div className="p-6 md:p-8 rounded-3xl bg-[#141418] border border-white/[0.08]">
          <span className="text-xs text-[#A3A3A8] uppercase tracking-wider block mb-2">
            Overall Accuracy
          </span>
          <span className="text-4xl md:text-5xl font-bold text-[#F4F3F0]">97.30%</span>
          <span className="text-[11px] text-[#54D6A0] block mt-2">
            865 of 889 Correct
          </span>
        </div>

        <div className="p-6 md:p-8 rounded-3xl bg-[#141418] border border-white/[0.08]">
          <span className="text-xs text-[#A3A3A8] uppercase tracking-wider block mb-2">
            Spam Precision
          </span>
          <span className="text-4xl md:text-5xl font-bold text-[#F4F3F0]">97.33%</span>
          <span className="text-[11px] text-[#A3A3A8] block mt-2">
            255 TP / 262 Flagged
          </span>
        </div>

        <div className="p-6 md:p-8 rounded-3xl bg-[#141418] border border-white/[0.08]">
          <span className="text-xs text-[#A3A3A8] uppercase tracking-wider block mb-2">
            Spam Recall
          </span>
          <span className="text-4xl md:text-5xl font-bold text-[#F4F3F0]">93.75%</span>
          <span className="text-[11px] text-[#A3A3A8] block mt-2">
            255 TP / 272 Actual
          </span>
        </div>

        <div className="p-6 md:p-8 rounded-3xl bg-[#141418] border border-[#B8F44A]/30 bg-gradient-to-br from-[#141418] to-[#1a2414]">
          <span className="text-xs text-[#B8F44A] uppercase tracking-wider block mb-2">
            Area Under ROC (AUC)
          </span>
          <span className="text-4xl md:text-5xl font-bold text-[#B8F44A]">0.9965</span>
          <span className="text-[11px] text-[#B8F44A] block mt-2">
            Near-Perfect Discrimination
          </span>
        </div>
      </div>

      {/* Two Column Section: Interactive Confusion Matrix & Baseline Showdown */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 my-16">
        {/* Left: Empirical Confusion Matrix */}
        <div className="lg:col-span-6 bg-[#141418] border border-white/[0.08] rounded-3xl p-8 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between pb-4 border-b border-white/[0.08] mb-6">
              <span className="font-mono text-xs uppercase text-[#A3A3A8]">
                Contingency Matrix (n=889)
              </span>
              <span className="font-mono text-xs text-[#B8F44A]">
                SPECIFICITY: 98.87%
              </span>
            </div>

            <h3 className="text-2xl font-title text-[#F4F3F0] mb-6">
              Held-Out Test Set Confusion Matrix
            </h3>

            {/* Matrix 2x2 Grid */}
            <div className="grid grid-cols-2 gap-4 font-mono text-center">
              <div className="p-6 rounded-2xl bg-[#0B0B0D] border border-[#54D6A0]/30 flex flex-col items-center justify-center">
                <span className="text-xs text-[#54D6A0] uppercase font-semibold mb-1">
                  True Negative (TN)
                </span>
                <span className="text-3xl font-bold text-[#F4F3F0]">610</span>
                <span className="text-[10px] text-[#A3A3A8] mt-1">
                  Ham Correctly Cleared (68.62%)
                </span>
              </div>

              <div className="p-6 rounded-2xl bg-[#0B0B0D] border border-[#FF5D66]/40 flex flex-col items-center justify-center">
                <span className="text-xs text-[#FF5D66] uppercase font-semibold mb-1">
                  False Positive (FP)
                </span>
                <span className="text-3xl font-bold text-[#FF5D66]">7</span>
                <span className="text-[10px] text-[#A3A3A8] mt-1">
                  Type I Error Rate: 1.13%
                </span>
              </div>

              <div className="p-6 rounded-2xl bg-[#0B0B0D] border border-[#f59e0b]/40 flex flex-col items-center justify-center">
                <span className="text-xs text-[#f59e0b] uppercase font-semibold mb-1">
                  False Negative (FN)
                </span>
                <span className="text-3xl font-bold text-[#f59e0b]">17</span>
                <span className="text-[10px] text-[#A3A3A8] mt-1">
                  Type II Error Rate: 6.25%
                </span>
              </div>

              <div className="p-6 rounded-2xl bg-[#0B0B0D] border border-[#54D6A0]/30 flex flex-col items-center justify-center">
                <span className="text-xs text-[#54D6A0] uppercase font-semibold mb-1">
                  True Positive (TP)
                </span>
                <span className="text-3xl font-bold text-[#F4F3F0]">255</span>
                <span className="text-[10px] text-[#A3A3A8] mt-1">
                  Spam Quarantined (28.68%)
                </span>
              </div>
            </div>
          </div>

          <p className="mt-8 pt-4 border-t border-white/[0.06] text-xs text-[#A3A3A8] font-light leading-relaxed">
            Out of 617 authentic legitimate emails in the untouched test partition, exactly 7 were misclassified (98.87% clean inbox retention).
          </p>
        </div>

        {/* Right: Model vs Majority Baseline Comparison */}
        <div className="lg:col-span-6 bg-[#141418] border border-white/[0.08] rounded-3xl p-8 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between pb-4 border-b border-white/[0.08] mb-6">
              <span className="font-mono text-xs uppercase text-[#A3A3A8]">
                Benchmark Competency CO5
              </span>
              <span className="font-mono text-xs text-[#FF5D66]">
                BASELINE FLAW EXPOSED
              </span>
            </div>

            <h3 className="text-2xl font-title text-[#F4F3F0] mb-3">
              Why Accuracy Alone Is Deceptive
            </h3>

            <p className="text-sm text-[#A3A3A8] font-light leading-relaxed mb-6">
              Due to natural class imbalance (69.37% Ham), a dummy majority-class model predicting &quot;Ham&quot; for every email scores 69.40% Accuracy despite having 0% utility.
            </p>

            <div className="space-y-4 font-mono text-xs">
              <div className="p-4 rounded-xl bg-[#0B0B0D] border border-white/[0.06]">
                <div className="flex justify-between mb-2">
                  <span className="text-[#F4F3F0]">SpamGuard Multinomial NB</span>
                  <span className="text-[#B8F44A] font-bold">F1: 95.51% • ACC: 97.30%</span>
                </div>
                <div className="w-full h-2 bg-white/[0.08] rounded-full overflow-hidden">
                  <div className="w-[97.3%] h-full bg-[#B8F44A] rounded-full" />
                </div>
              </div>

              <div className="p-4 rounded-xl bg-[#0B0B0D] border border-white/[0.06]">
                <div className="flex justify-between mb-2">
                  <span className="text-[#A3A3A8]">Dummy Majority Baseline</span>
                  <span className="text-[#FF5D66] font-bold">F1: 0.00% • ACC: 69.40%</span>
                </div>
                <div className="w-full h-2 bg-white/[0.08] rounded-full overflow-hidden">
                  <div className="w-[69.4%] h-full bg-[#FF5D66]/50 rounded-full" />
                </div>
                <span className="text-[10px] text-[#A3A3A8] block mt-2">
                  Catches 0 out of 272 spam emails. Total operational failure.
                </span>
              </div>
            </div>
          </div>

          <div className="mt-6 pt-4 border-t border-white/[0.06] font-mono text-xs flex justify-between text-[#A3A3A8]">
            <span>DELTA GAIN OVER BASELINE:</span>
            <span className="text-[#54D6A0] font-bold">+27.90% ACCURACY / +95.51% F1</span>
          </div>
        </div>
      </div>

      {/* Verified Publication Visualizations Showcase */}
      <div className="mt-16">
        <div className="flex items-center justify-between pb-6 border-b border-white/[0.08] mb-8">
          <span className="font-mono text-xs uppercase tracking-wider text-[#A3A3A8]">
            Official Academic Figures (reports/figures/)
          </span>
          <span className="font-mono text-xs text-[#B8F44A]">
            CLICK TO EXPAND HIGH-RES
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {figures.map((fig) => (
            <div
              key={fig.filename}
              onClick={() => setSelectedFigure(fig.filename)}
              className="group bg-[#141418] border border-white/[0.08] hover:border-white/[0.25] rounded-2xl overflow-hidden cursor-pointer transition-all duration-300 flex flex-col justify-between"
            >
              <div className="relative aspect-[16/10] bg-[#0B0B0D] overflow-hidden">
                <Image
                  src={`/figures/${fig.filename}`}
                  alt={fig.title}
                  fill
                  className="object-contain p-2 group-hover:scale-105 transition-transform duration-500"
                />
                <div className="absolute inset-0 bg-[#0B0B0D]/30 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                  <span className="p-2 rounded-full bg-[#B8F44A] text-[#0B0B0D]">
                    <Maximize2 className="w-4 h-4" />
                  </span>
                </div>
              </div>

              <div className="p-5 flex flex-col justify-between">
                <div>
                  <h4 className="text-base font-semibold text-[#F4F3F0] group-hover:text-[#B8F44A] transition-colors mb-1">
                    {fig.title}
                  </h4>
                  <p className="text-xs text-[#A3A3A8] font-light leading-relaxed">
                    {fig.caption}
                  </p>
                </div>
                <div className="mt-3 pt-3 border-t border-white/[0.04] flex items-center justify-between font-mono text-[10px] text-[#66666E]">
                  <span>PNG ARTIFACT</span>
                  <span>VERIFIED PHASE 06</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Lightbox Modal */}
      {selectedFigure && (
        <div
          onClick={() => setSelectedFigure(null)}
          className="fixed inset-0 z-50 bg-[#0B0B0D]/90 backdrop-blur-md flex items-center justify-center p-6"
        >
          <div
            onClick={(e) => e.stopPropagation()}
            className="max-w-4xl w-full bg-[#141418] border border-white/[0.15] rounded-3xl p-6 relative flex flex-col items-center shadow-2xl"
          >
            <button
              onClick={() => setSelectedFigure(null)}
              className="absolute top-4 right-4 text-[#A3A3A8] hover:text-[#F4F3F0] font-mono text-sm px-3 py-1 rounded-full bg-white/[0.05]"
            >
              CLOSE [ESC]
            </button>
            <div className="relative w-full aspect-[16/10] my-4 bg-[#0B0B0D] rounded-2xl overflow-hidden">
              <Image
                src={`/figures/${selectedFigure}`}
                alt="Selected Figure"
                fill
                className="object-contain p-4"
              />
            </div>
            <span className="font-mono text-xs text-[#A3A3A8]">
              Artifact: reports/figures/{selectedFigure}
            </span>
          </div>
        </div>
      )}
    </section>
  );
}
