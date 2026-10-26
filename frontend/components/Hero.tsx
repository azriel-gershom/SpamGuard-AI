"use client";

import React, { useEffect, useRef } from "react";
import { ArrowDownRight, Sparkles, Terminal, Activity, ArrowRight } from "lucide-react";

export default function Hero() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animationFrameId: number;
    let width = (canvas.width = canvas.parentElement?.clientWidth || 800);
    let height = (canvas.height = canvas.parentElement?.clientHeight || 450);

    const handleResize = () => {
      if (!canvas || !canvas.parentElement) return;
      width = canvas.width = canvas.parentElement.clientWidth;
      height = canvas.height = canvas.parentElement.clientHeight;
    };
    window.addEventListener("resize", handleResize);

    // Particle nodes representing email streams
    interface Particle {
      x: number;
      y: number;
      vx: number;
      vy: number;
      size: number;
      isThreat: boolean;
      opacity: number;
      char: string;
    }

    const chars = ["@", "#", "!", "$", "%", "0", "1", "<", ">", "/", "*"];
    const particles: Particle[] = Array.from({ length: 45 }, () => ({
      x: Math.random() * width,
      y: Math.random() * height,
      vx: (Math.random() - 0.5) * 0.45,
      vy: (Math.random() - 0.5) * 0.45,
      size: Math.random() * 2 + 1.5,
      isThreat: Math.random() < 0.31, // exact 30.63% dataset ratio
      opacity: Math.random() * 0.5 + 0.2,
      char: chars[Math.floor(Math.random() * chars.length)],
    }));

    const render = () => {
      ctx.clearRect(0, 0, width, height);

      // Connect nodes within proximity
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);

          if (dist < 110) {
            ctx.beginPath();
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            const bothThreat = particles[i].isThreat && particles[j].isThreat;
            ctx.strokeStyle = bothThreat
              ? `rgba(255, 93, 102, ${0.15 * (1 - dist / 110)})`
              : `rgba(184, 244, 74, ${0.12 * (1 - dist / 110)})`;
            ctx.lineWidth = 0.75;
            ctx.stroke();
          }
        }
      }

      // Draw particle badges
      particles.forEach((p) => {
        p.x += p.vx;
        p.y += p.vy;

        if (p.x < 0) p.x = width;
        if (p.x > width) p.x = 0;
        if (p.y < 0) p.y = height;
        if (p.y > height) p.y = 0;

        ctx.fillStyle = p.isThreat ? "rgba(255, 93, 102, 0.7)" : "rgba(84, 214, 160, 0.7)";
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fill();

        ctx.font = "9px monospace";
        ctx.fillStyle = p.isThreat ? "rgba(255, 93, 102, 0.4)" : "rgba(244, 243, 240, 0.25)";
        ctx.fillText(p.char, p.x + 6, p.y + 3);
      });

      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => {
      window.removeEventListener("resize", handleResize);
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  return (
    <section
      id="overview"
      className="relative pt-32 pb-16 md:pt-44 md:pb-24 px-6 md:px-12 max-w-7xl mx-auto flex flex-col justify-between min-h-[740px] md:min-h-[820px]"
    >
      {/* Background Interactive Stream */}
      <div className="absolute inset-0 z-0 pointer-events-none opacity-60 overflow-hidden">
        <canvas ref={canvasRef} className="w-full h-full block" />
      </div>

      {/* Top Metadata Strip */}
      <div className="relative z-10 flex flex-wrap items-center justify-between gap-4 font-mono text-[11px] uppercase tracking-[0.14em] text-[#A3A3A8] pb-8 border-b border-white/[0.08]">
        <div className="flex items-center gap-3">
          <span className="w-2 h-2 rounded-full bg-[#B8F44A]"></span>
          <span>RESEARCH INITIATIVE / SEM 3 AI MICRO PROJECT – 05</span>
        </div>
        <div className="flex items-center gap-4">
          <span className="hidden sm:inline">FROZEN PIPELINE SHA-256: C652119...</span>
          <span className="text-[#B8F44A] bg-[#B8F44A]/10 px-2.5 py-0.5 rounded border border-[#B8F44A]/30">
            97.30% VERIFIED ACCURACY
          </span>
        </div>
      </div>

      {/* Main Giant Headline */}
      <div className="relative z-10 my-12 md:my-16">
        <div className="max-w-5xl">
          <h1 className="text-5xl sm:text-7xl md:text-8xl lg:text-9xl font-title tracking-tight text-[#F4F3F0] uppercase leading-[0.92] select-none">
            Your inbox. <br />
            Without the <br />
            <span className="italic font-normal text-[#B8F44A] tracking-normal">
              Noise.
            </span>
          </h1>

          <p className="mt-8 text-lg sm:text-xl md:text-2xl text-[#A3A3A8] max-w-2xl font-sans font-light leading-relaxed">
            Intelligent email classification. Powered by explainable machine learning.
            Sub-millisecond inference separating legitimate correspondence from malicious deception.
          </p>
        </div>

        {/* Action CTAs */}
        <div className="mt-10 flex flex-wrap items-center gap-5">
          <a
            href="#scanner"
            className="group px-7 py-4 rounded-full bg-[#B8F44A] text-[#0B0B0D] font-mono text-xs uppercase tracking-wider font-semibold hover:bg-[#c9f86f] transition-all duration-300 shadow-lg shadow-[#B8F44A]/20 flex items-center gap-2 hover:translate-x-1"
          >
            <span>Launch Scanner</span>
            <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
          </a>

          <a
            href="#analytics"
            className="px-7 py-4 rounded-full bg-[#141418] border border-white/[0.12] text-[#F4F3F0] font-mono text-xs uppercase tracking-wider hover:bg-white/[0.05] hover:border-white/[0.25] transition-all duration-300 flex items-center gap-2"
          >
            <span>Explore the Intelligence</span>
            <ArrowDownRight className="w-4 h-4 text-[#A3A3A8]" />
          </a>
        </div>
      </div>

      {/* Bottom Editorial Stats Strip */}
      <div className="relative z-10 grid grid-cols-2 md:grid-cols-4 gap-6 pt-10 border-t border-white/[0.08] font-mono">
        <div className="flex flex-col">
          <span className="text-3xl md:text-4xl text-[#F4F3F0] font-semibold tracking-tight">
            97.30%
          </span>
          <span className="text-[11px] text-[#A3A3A8] uppercase tracking-wider mt-1">
            Test Set Accuracy
          </span>
        </div>

        <div className="flex flex-col">
          <span className="text-3xl md:text-4xl text-[#54D6A0] font-semibold tracking-tight">
            95.51%
          </span>
          <span className="text-[11px] text-[#A3A3A8] uppercase tracking-wider mt-1">
            Spam F1-Score
          </span>
        </div>

        <div className="flex flex-col">
          <span className="text-3xl md:text-4xl text-[#F4F3F0] font-semibold tracking-tight">
            889
          </span>
          <span className="text-[11px] text-[#A3A3A8] uppercase tracking-wider mt-1">
            Held-Out Test Emails
          </span>
        </div>

        <div className="flex flex-col">
          <span className="text-3xl md:text-4xl text-[#B8F44A] font-semibold tracking-tight">
            1.04 ms
          </span>
          <span className="text-[11px] text-[#A3A3A8] uppercase tracking-wider mt-1">
            Warm Median Latency
          </span>
        </div>
      </div>
    </section>
  );
}
