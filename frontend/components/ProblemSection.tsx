"use client";

import React, { useState } from "react";
import { AlertTriangle, CheckCircle2, ShieldAlert, Inbox, Filter } from "lucide-react";

export default function ProblemSection() {
  const [filterMode, setFilterMode] = useState<"all" | "ham" | "spam">("all");

  const sampleInbox = [
    {
      id: 1,
      type: "spam",
      sender: "Security Team <alert@security-update-notice.com>",
      subject: "URGENT: Unauthorized access detected on your banking account",
      preview: "We have detected suspicious login attempts. Verify credentials within 24 hours to prevent termination...",
      tag: "Phishing Threat",
      latency: "0.98 ms",
    },
    {
      id: 2,
      type: "ham",
      sender: "Alex Chen <alex.chen@engineering-corp.org>",
      subject: "Sprint 34 Standup Notes & Feature Branch Code Review",
      preview: "Hi team, please find attached the meeting notes and git patch for today's review...",
      tag: "Verified Ham",
      latency: "1.05 ms",
    },
    {
      id: 3,
      type: "spam",
      sender: "Global Rewards <winner@promotional-claim-bonus.net>",
      subject: "Congratulations! Claim your free cash prize bonus money now",
      preview: "You have been selected as the guaranteed winner of our exclusive $10,000 cash sweepstakes prize...",
      tag: "Commercial Scam",
      latency: "1.12 ms",
    },
    {
      id: 4,
      type: "ham",
      sender: "Course Coordinator <eval@academic-faculty.edu>",
      subject: "AI Micro Project 05: SpamGuard AI Documentation & Report",
      preview: "Please find attached the final project documentation and evaluation metrics for Micro Project 05...",
      tag: "Verified Ham",
      latency: "1.01 ms",
    },
  ];

  const visibleItems = sampleInbox.filter((item) => {
    if (filterMode === "all") return true;
    return item.type === filterMode;
  });

  return (
    <section className="py-24 px-6 md:px-12 max-w-7xl mx-auto border-t border-white/[0.08]">
      {/* Category Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-6 pb-12 border-b border-white/[0.08]">
        <div>
          <span className="font-mono text-xs uppercase tracking-[0.14em] text-[#A3A3A8] block mb-2">
            01 / The Threat Landscape
          </span>
          <h2 className="text-4xl sm:text-6xl font-title uppercase tracking-tight text-[#F4F3F0]">
            The Cluttered <span className="italic text-[#FF5D66]">Threat</span> Vector
          </h2>
        </div>
        <p className="max-w-md text-sm md:text-base text-[#A3A3A8] font-light leading-relaxed">
          Over 45% of global email volume is unsolicited commercial and adversarial traffic.
          Traditional rule-based regex filters collapse when faced with obfuscation, synonym swapping, and subtle social engineering.
        </p>
      </div>

      {/* Two Column Narrative: The Imbalance & The Cost of Errors */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 my-16">
        {/* Left Column: Asymmetric Error Cost */}
        <div className="lg:col-span-5 bg-[#141418] border border-white/[0.08] rounded-2xl p-8 flex flex-col justify-between">
          <div>
            <div className="w-10 h-10 rounded-full bg-[#FF5D66]/10 border border-[#FF5D66]/30 flex items-center justify-center text-[#FF5D66] mb-6">
              <ShieldAlert className="w-5 h-5" />
            </div>
            <h3 className="text-2xl font-title text-[#F4F3F0] mb-3">
              The Asymmetry of Classification Error
            </h3>
            <p className="text-sm text-[#A3A3A8] leading-relaxed mb-6 font-light">
              In email security, the operational cost of errors is heavily asymmetrical:
            </p>

            <div className="space-y-4">
              <div className="p-4 rounded-xl bg-[#0B0B0D] border border-white/[0.06]">
                <div className="flex items-center justify-between text-xs font-mono uppercase mb-1">
                  <span className="text-[#FF5D66]">Type I Error (False Positive)</span>
                  <span className="text-[#54D6A0]">1.13% In SpamGuard</span>
                </div>
                <p className="text-xs text-[#A3A3A8]">
                  Quarantining a critical legitimate email (e.g., job offer, banking reset, client contract). Extremely costly to human operations.
                </p>
              </div>

              <div className="p-4 rounded-xl bg-[#0B0B0D] border border-white/[0.06]">
                <div className="flex items-center justify-between text-xs font-mono uppercase mb-1">
                  <span className="text-[#B8F44A]">Type II Error (False Negative)</span>
                  <span className="text-[#A3A3A8]">6.25% In SpamGuard</span>
                </div>
                <p className="text-xs text-[#A3A3A8]">
                  A spam email leaking into the user inbox. Annoying, but recoverable via downstream secondary defenses.
                </p>
              </div>
            </div>
          </div>

          <div className="mt-8 pt-6 border-t border-white/[0.06] flex items-center justify-between font-mono text-xs">
            <span className="text-[#A3A3A8]">NATURAL DATASET RATIO:</span>
            <span className="text-[#F4F3F0] font-semibold">2.27 HAM : 1 SPAM</span>
          </div>
        </div>

        {/* Right Column: Interactive Inbox Demonstration */}
        <div className="lg:col-span-7 bg-[#141418] border border-white/[0.08] rounded-2xl p-8 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between pb-6 border-b border-white/[0.08] mb-6">
              <div className="flex items-center gap-3">
                <Inbox className="w-5 h-5 text-[#B8F44A]" />
                <span className="font-mono text-xs uppercase tracking-wider text-[#F4F3F0]">
                  Live Stream Filtering Demonstration
                </span>
              </div>

              {/* Filter Chips */}
              <div className="flex items-center gap-1.5 font-mono text-[11px] uppercase">
                <button
                  onClick={() => setFilterMode("all")}
                  className={`px-3 py-1 rounded-full transition-all ${
                    filterMode === "all"
                      ? "bg-[#F4F3F0] text-[#0B0B0D] font-semibold"
                      : "text-[#A3A3A8] hover:text-[#F4F3F0] bg-[#0B0B0D]"
                  }`}
                >
                  All (4)
                </button>
                <button
                  onClick={() => setFilterMode("ham")}
                  className={`px-3 py-1 rounded-full transition-all ${
                    filterMode === "ham"
                      ? "bg-[#54D6A0] text-[#0B0B0D] font-semibold"
                      : "text-[#A3A3A8] hover:text-[#54D6A0] bg-[#0B0B0D]"
                  }`}
                >
                  Ham Only
                </button>
                <button
                  onClick={() => setFilterMode("spam")}
                  className={`px-3 py-1 rounded-full transition-all ${
                    filterMode === "spam"
                      ? "bg-[#FF5D66] text-[#0B0B0D] font-semibold"
                      : "text-[#A3A3A8] hover:text-[#FF5D66] bg-[#0B0B0D]"
                  }`}
                >
                  Spam Only
                </button>
              </div>
            </div>

            {/* Email Row List with Anthracite Hover Dimming */}
            <div className="editorial-group space-y-3">
              {visibleItems.map((item) => (
                <div
                  key={item.id}
                  className="editorial-row p-4 rounded-xl bg-[#0B0B0D] border border-white/[0.06] hover:border-white/[0.2] transition-all duration-300 flex flex-col gap-2 cursor-default"
                >
                  <div className="flex items-center justify-between text-xs font-mono">
                    <span className="text-[#A3A3A8] truncate max-w-[260px]">
                      {item.sender}
                    </span>
                    <span
                      className={`px-2 py-0.5 rounded text-[10px] uppercase font-semibold tracking-wider ${
                        item.type === "spam"
                          ? "bg-[#FF5D66]/15 text-[#FF5D66] border border-[#FF5D66]/30"
                          : "bg-[#54D6A0]/15 text-[#54D6A0] border border-[#54D6A0]/30"
                      }`}
                    >
                      {item.tag}
                    </span>
                  </div>

                  <h4 className="text-sm font-semibold text-[#F4F3F0] truncate">
                    {item.subject}
                  </h4>

                  <p className="text-xs text-[#A3A3A8] font-light line-clamp-2">
                    {item.preview}
                  </p>

                  <div className="flex items-center justify-between pt-2 border-t border-white/[0.04] text-[10px] font-mono text-[#66666E]">
                    <span>INFERENCE: {item.latency}</span>
                    <span>MODEL: MNB + TF-IDF</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="mt-6 text-center text-xs text-[#A3A3A8] font-mono">
            Hover individual entries to experience interactive editorial focus isolation.
          </div>
        </div>
      </div>
    </section>
  );
}
