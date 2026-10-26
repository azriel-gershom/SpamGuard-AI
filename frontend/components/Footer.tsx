"use client";

import React from "react";
import { Shield, ArrowUp, ExternalLink, Terminal, GitBranch } from "lucide-react";

export default function Footer() {
  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  return (
    <footer className="border-t border-white/[0.08] bg-[#070709] py-16 px-6 md:px-12 text-[#A3A3A8] font-mono text-xs">
      <div className="max-w-7xl mx-auto flex flex-col gap-12">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-6 pb-12 border-b border-white/[0.06]">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-full bg-[#B8F44A]/10 border border-[#B8F44A]/30 flex items-center justify-center text-[#B8F44A]">
              <Shield className="w-4 h-4" />
            </div>
            <div>
              <span className="text-[#F4F3F0] font-bold text-sm tracking-wider uppercase block">
                SpamGuard AI
              </span>
              <span className="text-[10px] text-[#A3A3A8]">
                Anthracite-Inspired Threat Intelligence Interface
              </span>
            </div>
          </div>

          <div className="flex items-center gap-6">
            <a
              href="http://localhost:8501"
              target="_blank"
              rel="noopener noreferrer"
              className="text-[#A3A3A8] hover:text-[#B8F44A] flex items-center gap-1.5 transition-colors"
            >
              <span>Streamlit App (Port 8501)</span>
              <ExternalLink className="w-3.5 h-3.5" />
            </a>

            <button
              onClick={scrollToTop}
              className="p-2 rounded-full bg-white/[0.04] border border-white/[0.08] text-[#F4F3F0] hover:bg-[#B8F44A] hover:text-[#0B0B0D] transition-colors"
              aria-label="Back to top"
            >
              <ArrowUp className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Cryptographic Ledger & Metadata */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-[11px] leading-relaxed">
          <div>
            <span className="text-[#F4F3F0] uppercase font-bold block mb-2">
              Cryptographic Integrity
            </span>
            <p className="text-[#66666E]">
              Frozen Pipeline SHA-256 Checksum:
            </p>
            <p className="text-[#B8F44A] break-all select-all mt-1">
              c652119c99f0db2abe0788969d002818478bb63be95e31e5f337eda8da6ddbb0
            </p>
          </div>

          <div>
            <span className="text-[#F4F3F0] uppercase font-bold block mb-2">
              Academic Micro-Project
            </span>
            <p className="text-[#66666E]">
              Semester 3 Artificial Intelligence Micro Project – 05
            </p>
            <p className="mt-1">
              Course Competencies: CO2 (Algorithm), CO4 (Pipeline), CO5 (Metrics)
            </p>
          </div>

          <div>
            <span className="text-[#F4F3F0] uppercase font-bold block mb-2">
              Software Version Control
            </span>
            <p className="text-[#66666E]">
              Commit Baseline: <span className="text-[#F4F3F0]">6b84ea5</span>
            </p>
            <p className="mt-1">
              Automated Baseline: <span className="text-[#54D6A0]">131 Passing Tests</span>
            </p>
          </div>
        </div>

        <div className="pt-8 border-t border-white/[0.04] flex flex-col sm:flex-row items-center justify-between text-[10px] text-[#66666E] gap-4">
          <span>&copy; 2026 SpamGuard AI Research Group. All rights reserved.</span>
          <span>Designed with high-density editorial principles inspired by Anthracite Studio.</span>
        </div>
      </div>
    </footer>
  );
}
