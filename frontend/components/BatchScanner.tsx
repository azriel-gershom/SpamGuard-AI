"use client";

import React, { useState, useRef } from "react";
import {
  UploadCloud,
  FileSpreadsheet,
  Download,
  CheckCircle2,
  AlertTriangle,
  Filter,
  FileDown,
  ArrowUpDown,
  Search,
  Check,
  Clock,
} from "lucide-react";
import { classifyBatchCsv, BatchResponse, BatchItemResult } from "@/lib/api";

export default function BatchScanner() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<BatchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<"all" | "spam" | "ham">("all");
  const [searchQuery, setSearchQuery] = useState("");
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleFileDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const dropped = e.dataTransfer.files[0];
      if (dropped.name.toLowerCase().endsWith(".csv")) {
        setFile(dropped);
        setResult(null);
        setError(null);
      } else {
        setError("Invalid file format. Please upload a .csv file.");
      }
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const selected = e.target.files[0];
      if (selected.name.toLowerCase().endsWith(".csv")) {
        setFile(selected);
        setResult(null);
        setError(null);
      } else {
        setError("Invalid file format. Please upload a .csv file.");
      }
    }
  };

  const handleDownloadSample = () => {
    const sampleCsv =
      "subject,body\n" +
      '"Urgent Account Verification","We detected unauthorized logins from an unfamiliar IP address. Click here immediately to verify: http://phishing-scam-test.com/login"\n' +
      '"Sprint Standup Notes","Hi team, attached is the sprint backlog and documentation patch for Friday review."\n' +
      '"Congratulations Cash Winner","You won our $5,000 cash sweepstakes prize! Claim your immediate wire transfer now."\n' +
      '"AI Microproject Documentation","Respected Professor, please find attached the model evaluation and research report."\n' +
      '"Lower Mortgage Rates Today","Refinance your mortgage with zero fees and guaranteed approval today!"\n';

    const blob = new Blob([sampleCsv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "spamguard_sample_template.csv";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleScanBatch = async () => {
    if (!file) {
      setError("Please select or drop a CSV file first.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const data = await classifyBatchCsv(file, false);
      setResult(data);
    } catch (err: any) {
      setError(err.message || "Failed to process batch CSV.");
    } finally {
      setLoading(false);
    }
  };

  const handleExportResults = () => {
    if (!result || !result.items) return;

    // Sanitize headers and cells against CSV formula injection
    const header = "id,subject,body_preview,prediction,is_spam,spam_probability,ham_probability,latency_ms\n";
    const rows = result.items.map((item) => {
      const cleanSubj = `"${item.subject.replace(/"/g, '""')}"`;
      const cleanPrev = `"${item.body_preview.replace(/"/g, '""')}"`;
      return `${item.id},${cleanSubj},${cleanPrev},${item.predicted_label},${item.is_spam},${item.spam_probability},${item.ham_probability},${item.processing_time_ms}`;
    });

    const fullCsv = header + rows.join("\n");
    const blob = new Blob([fullCsv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `spamguard_classified_results_${Date.now()}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // Filter items
  const filteredItems = result
    ? result.items.filter((item) => {
        if (filter === "spam" && !item.is_spam) return false;
        if (filter === "ham" && item.is_spam) return false;
        if (searchQuery.trim()) {
          const q = searchQuery.toLowerCase();
          return (
            item.subject.toLowerCase().includes(q) ||
            item.body_preview.toLowerCase().includes(q)
          );
        }
        return true;
      })
    : [];

  return (
    <section id="batch" className="py-24 px-6 md:px-12 max-w-7xl mx-auto border-t border-white/[0.08]">
      {/* Category Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-6 pb-12 border-b border-white/[0.08]">
        <div>
          <span className="font-mono text-xs uppercase tracking-[0.14em] text-[#B8F44A] block mb-2">
            04 / Vectorized Throughput
          </span>
          <h2 className="text-4xl sm:text-6xl font-title uppercase tracking-tight text-[#F4F3F0]">
            Batch CSV <span className="italic text-[#B8F44A]">Classifier</span>
          </h2>
        </div>
        <div className="flex flex-col items-start md:items-end gap-3">
          <p className="max-w-md text-sm md:text-base text-[#A3A3A8] font-light leading-relaxed">
            High-throughput pipeline processing up to 10,000 records. Protected against spreadsheet formula injection (`=, +, -, @`).
          </p>
          <button
            onClick={handleDownloadSample}
            className="flex items-center gap-1.5 font-mono text-xs text-[#B8F44A] hover:underline"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Download Sample CSV Template</span>
          </button>
        </div>
      </div>

      {/* Upload Box & Action Strip */}
      <div className="my-12">
        <div
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleFileDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`border-2 border-dashed rounded-3xl p-10 md:p-14 text-center cursor-pointer transition-all duration-300 flex flex-col items-center justify-center ${
            file
              ? "bg-[#141418] border-[#B8F44A]/50"
              : "bg-[#0B0B0D] border-white/[0.1] hover:border-white/[0.25] hover:bg-[#141418]/50"
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv"
            onChange={handleFileChange}
            className="hidden"
          />

          <div className="w-16 h-16 rounded-full bg-white/[0.04] border border-white/[0.08] flex items-center justify-center text-[#B8F44A] mb-4">
            {file ? <FileSpreadsheet className="w-8 h-8" /> : <UploadCloud className="w-8 h-8" />}
          </div>

          <h3 className="text-xl font-title text-[#F4F3F0] mb-2">
            {file ? file.name : "Drag and drop your email dataset CSV"}
          </h3>

          <p className="text-xs text-[#A3A3A8] font-light max-w-sm mb-4">
            {file
              ? `${(file.size / 1024).toFixed(1)} KB — Ready for vectorized inference`
              : "Supports comma-separated files with 'subject' and 'body' columns. Up to 10,000 records."}
          </p>

          <span className="font-mono text-[11px] text-[#B8F44A] bg-[#B8F44A]/10 px-3 py-1 rounded-full border border-[#B8F44A]/20">
            {file ? "CLICK TO CHANGE FILE" : "BROWSE LOCAL FILES"}
          </span>
        </div>

        {error && (
          <div className="mt-4 p-4 rounded-xl bg-[#FF5D66]/10 border border-[#FF5D66]/30 text-[#FF5D66] text-xs flex items-center gap-3 font-mono">
            <AlertTriangle className="w-4 h-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Scan Button Trigger */}
        <div className="mt-6 flex flex-wrap items-center justify-between gap-4">
          <div className="font-mono text-xs text-[#A3A3A8]">
            BOUNDARY LIMIT: 10,000 ROWS • FORMULA INJECTION DEFENSE ARMED
          </div>

          <button
            onClick={handleScanBatch}
            disabled={!file || loading}
            className="px-8 py-3.5 rounded-full bg-[#B8F44A] text-[#0B0B0D] font-mono text-xs uppercase tracking-wider font-semibold hover:bg-[#c9f86f] transition-all shadow-lg shadow-[#B8F44A]/20 flex items-center gap-2 disabled:opacity-40 disabled:cursor-not-allowed hover:scale-[1.02]"
          >
            {loading ? (
              <>
                <span className="w-3.5 h-3.5 rounded-full border-2 border-[#0B0B0D] border-t-transparent animate-spin"></span>
                <span>Executing Vectorized Batch...</span>
              </>
            ) : (
              <>
                <Check className="w-4 h-4" />
                <span>Process Entire Batch</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Batch Results Output */}
      {result && (
        <div className="mt-12 bg-[#141418] border border-white/[0.08] rounded-3xl p-6 sm:p-10">
          {/* Summary KPIs Strip */}
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4 pb-8 border-b border-white/[0.08] font-mono">
            <div className="p-4 rounded-2xl bg-[#0B0B0D] border border-white/[0.06]">
              <span className="text-[10px] uppercase text-[#A3A3A8] block">Total Ingested</span>
              <span className="text-2xl font-bold text-[#F4F3F0]">{result.total_count}</span>
            </div>

            <div className="p-4 rounded-2xl bg-[#0B0B0D] border border-[#FF5D66]/20">
              <span className="text-[10px] uppercase text-[#FF5D66] block">Spam Flagged</span>
              <span className="text-2xl font-bold text-[#FF5D66]">
                {result.spam_count} <span className="text-xs font-normal">({result.spam_percentage}%)</span>
              </span>
            </div>

            <div className="p-4 rounded-2xl bg-[#0B0B0D] border border-[#54D6A0]/20">
              <span className="text-[10px] uppercase text-[#54D6A0] block">Ham Cleared</span>
              <span className="text-2xl font-bold text-[#54D6A0]">
                {result.ham_count} <span className="text-xs font-normal">({result.ham_percentage}%)</span>
              </span>
            </div>

            <div className="p-4 rounded-2xl bg-[#0B0B0D] border border-white/[0.06]">
              <span className="text-[10px] uppercase text-[#A3A3A8] block">Total Duration</span>
              <span className="text-2xl font-bold text-[#F4F3F0]">{result.total_duration_ms} ms</span>
            </div>

            <div className="p-4 rounded-2xl bg-[#0B0B0D] border border-white/[0.06] col-span-2 md:col-span-1">
              <span className="text-[10px] uppercase text-[#B8F44A] block">Avg Per-Email</span>
              <span className="text-2xl font-bold text-[#B8F44A]">{result.avg_latency_ms} ms</span>
            </div>
          </div>

          {/* Table Controls (Filter & Export) */}
          <div className="my-6 flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-2">
              <span className="text-xs font-mono text-[#A3A3A8] uppercase">Filter:</span>
              <div className="flex gap-1.5 font-mono text-xs">
                <button
                  onClick={() => setFilter("all")}
                  className={`px-3 py-1 rounded-full ${
                    filter === "all"
                      ? "bg-[#F4F3F0] text-[#0B0B0D] font-bold"
                      : "bg-[#0B0B0D] text-[#A3A3A8] hover:text-[#F4F3F0]"
                  }`}
                >
                  All ({result.total_count})
                </button>
                <button
                  onClick={() => setFilter("spam")}
                  className={`px-3 py-1 rounded-full ${
                    filter === "spam"
                      ? "bg-[#FF5D66] text-[#0B0B0D] font-bold"
                      : "bg-[#0B0B0D] text-[#FF5D66] hover:bg-[#FF5D66]/20"
                  }`}
                >
                  Spam ({result.spam_count})
                </button>
                <button
                  onClick={() => setFilter("ham")}
                  className={`px-3 py-1 rounded-full ${
                    filter === "ham"
                      ? "bg-[#54D6A0] text-[#0B0B0D] font-bold"
                      : "bg-[#0B0B0D] text-[#54D6A0] hover:bg-[#54D6A0]/20"
                  }`}
                >
                  Ham ({result.ham_count})
                </button>
              </div>
            </div>

            <button
              onClick={handleExportResults}
              className="px-5 py-2 rounded-full bg-white/[0.06] border border-white/[0.1] text-xs font-mono uppercase text-[#F4F3F0] hover:bg-[#B8F44A] hover:text-[#0B0B0D] transition-colors flex items-center gap-2"
            >
              <FileDown className="w-4 h-4" />
              <span>Export Classified CSV</span>
            </button>
          </div>

          {/* Results Table */}
          <div className="overflow-x-auto rounded-2xl border border-white/[0.06]">
            <table className="w-full text-left font-mono text-xs">
              <thead className="bg-[#0B0B0D] border-b border-white/[0.08] text-[#A3A3A8] uppercase text-[10px]">
                <tr>
                  <th className="p-4">#</th>
                  <th className="p-4">Subject</th>
                  <th className="p-4">Content Preview</th>
                  <th className="p-4">Verdict</th>
                  <th className="p-4 text-right">Spam Prob</th>
                  <th className="p-4 text-right">Latency</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/[0.04]">
                {filteredItems.map((item, idx) => (
                  <tr
                    key={item.id || idx}
                    className="hover:bg-white/[0.02] transition-colors"
                  >
                    <td className="p-4 text-[#A3A3A8]">{item.id || idx + 1}</td>
                    <td className="p-4 text-[#F4F3F0] font-semibold max-w-[200px] truncate">
                      {item.subject}
                    </td>
                    <td className="p-4 text-[#A3A3A8] max-w-[320px] truncate">
                      {item.body_preview}
                    </td>
                    <td className="p-4">
                      <span
                        className={`px-2 py-0.5 rounded text-[10px] uppercase font-bold tracking-wider ${
                          item.is_spam
                            ? "bg-[#FF5D66]/20 text-[#FF5D66] border border-[#FF5D66]/40"
                            : "bg-[#54D6A0]/20 text-[#54D6A0] border border-[#54D6A0]/40"
                        }`}
                      >
                        {item.predicted_label}
                      </span>
                    </td>
                    <td
                      className={`p-4 text-right font-bold ${
                        item.is_spam ? "text-[#FF5D66]" : "text-[#54D6A0]"
                      }`}
                    >
                      {(item.spam_probability * 100).toFixed(2)}%
                    </td>
                    <td className="p-4 text-right text-[#A3A3A8]">
                      {item.processing_time_ms} ms
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </section>
  );
}
