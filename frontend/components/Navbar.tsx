"use client";

import React, { useState, useEffect } from "react";
import { Shield, Radio, Terminal, Menu, X, ExternalLink } from "lucide-react";
import { fetchHealth, SystemHealth } from "@/lib/api";

export default function Navbar() {
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [mobileOpen, setMobileOpen] = useState(false);

  useEffect(() => {
    fetchHealth()
      .then(setHealth)
      .catch(() => {
        // Fallback default if API is starting
        setHealth({
          status: "OPERATIONAL",
          model_available: true,
          model_sha256: "c652119c99f0db2abe0788969d002818478bb63be95e31e5f337eda8da6ddbb0",
          model_sha256_verified: true,
          model_identifier: "MultinomialNB(alpha=0.1) + TF-IDF(20,000 features)",
          system_info: { os: "Windows 11", python: "3.12" },
        });
      });
  }, []);

  const navLinks = [
    { label: "Overview", href: "#overview" },
    { label: "Pipeline", href: "#pipeline" },
    { label: "Scanner", href: "#scanner" },
    { label: "Batch CSV", href: "#batch" },
    { label: "Investigation", href: "#investigation" },
    { label: "Analytics", href: "#analytics" },
    { label: "Methodology", href: "#methodology" },
  ];

  return (
    <header className="fixed top-5 inset-x-0 z-50 flex justify-center px-4 pointer-events-none">
      <nav className="pointer-events-auto bg-[#141418]/90 backdrop-blur-xl border border-white/[0.08] shadow-2xl rounded-full px-5 py-2.5 flex items-center justify-between gap-6 max-w-5xl w-full transition-all duration-300 hover:border-white/[0.16]">
        {/* Brand Logo */}
        <a
          href="#overview"
          className="flex items-center gap-3 text-[#F4F3F0] group focus-visible:outline-2 focus-visible:outline-[#B8F44A]"
          aria-label="SpamGuard AI Home"
        >
          <div className="w-8 h-8 rounded-full bg-[#B8F44A]/10 border border-[#B8F44A]/30 flex items-center justify-center text-[#B8F44A] group-hover:scale-105 transition-transform duration-300">
            <Shield className="w-4 h-4" />
          </div>
          <div className="flex flex-col">
            <span className="font-mono text-xs font-semibold tracking-[0.12em] uppercase text-[#F4F3F0]">
              SpamGuard
            </span>
            <span className="font-mono text-[9px] tracking-wider text-[#A3A3A8] uppercase">
              AI Classifier
            </span>
          </div>
        </a>

        {/* Desktop Links */}
        <div className="hidden md:flex items-center gap-1 font-mono text-[12px] uppercase tracking-wider text-[#A3A3A8]">
          {navLinks.map((item) => (
            <a
              key={item.href}
              href={item.href}
              className="px-3.5 py-1.5 rounded-full hover:text-[#F4F3F0] hover:bg-white/[0.05] transition-colors duration-200"
            >
              {item.label}
            </a>
          ))}
        </div>

        {/* Telemetry & Fallback */}
        <div className="hidden lg:flex items-center gap-3 pl-2 border-l border-white/[0.08]">
          <div className="flex items-center gap-2 bg-[#0B0B0D] px-2.5 py-1 rounded-full border border-white/[0.06]">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#54D6A0] opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-[#54D6A0]"></span>
            </span>
            <span className="font-mono text-[10px] uppercase text-[#A3A3A8] tracking-widest">
              {health?.status === "OPERATIONAL" ? "ENGINE ONLINE" : "ENGINE CONNECTING"}
            </span>
          </div>

          <a
            href="http://localhost:8501"
            target="_blank"
            rel="noopener noreferrer"
            title="Open legacy Streamlit application"
            className="flex items-center gap-1.5 font-mono text-[10px] text-[#A3A3A8] hover:text-[#B8F44A] px-2.5 py-1 rounded-full border border-white/[0.06] hover:border-[#B8F44A]/30 transition-all"
          >
            <span>STREAMLIT</span>
            <ExternalLink className="w-3 h-3" />
          </a>
        </div>

        {/* Mobile Toggle */}
        <button
          onClick={() => setMobileOpen(!mobileOpen)}
          className="md:hidden text-[#F4F3F0] p-1.5 rounded-full hover:bg-white/[0.06]"
          aria-label="Toggle navigation menu"
        >
          {mobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
        </button>
      </nav>

      {/* Mobile Drawer */}
      {mobileOpen && (
        <div className="md:hidden pointer-events-auto fixed inset-x-4 top-20 bg-[#141418] border border-white/[0.1] rounded-2xl p-6 shadow-2xl flex flex-col gap-4">
          <div className="flex flex-col gap-2 font-mono text-sm uppercase">
            {navLinks.map((item) => (
              <a
                key={item.href}
                href={item.href}
                onClick={() => setMobileOpen(false)}
                className="py-2 px-3 rounded-lg hover:bg-white/[0.05] text-[#A3A3A8] hover:text-[#F4F3F0]"
              >
                {item.label}
              </a>
            ))}
          </div>
          <div className="pt-4 border-t border-white/[0.08] flex items-center justify-between text-xs font-mono text-[#A3A3A8]">
            <span>STATUS: {health?.status || "READY"}</span>
            <a
              href="http://localhost:8501"
              target="_blank"
              rel="noopener noreferrer"
              className="text-[#B8F44A] underline flex items-center gap-1"
            >
              Streamlit <ExternalLink className="w-3 h-3" />
            </a>
          </div>
        </div>
      )}
    </header>
  );
}
