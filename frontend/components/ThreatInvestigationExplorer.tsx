"use client";

import React, { useState, useEffect, useMemo } from "react";
import {
  Compass,
  Network,
  Mail,
  Globe,
  Link as LinkIcon,
  Search,
  ArrowRight,
  CheckCircle2,
  AlertCircle,
  RefreshCw,
  Plus,
  Trash2,
  Upload,
  Layers,
  HelpCircle,
  Eye,
  Info,
} from "lucide-react";
import {
  fetchSearchDemoEmails,
  runGreedyBestFirstSearch,
  InvestigationEmailItem,
  SearchResponse,
} from "@/lib/api";

export default function ThreatInvestigationExplorer() {
  const [emails, setEmails] = useState<InvestigationEmailItem[]>([]);
  const [startEmailId, setStartEmailId] = useState<string>("Email-1");
  const [goalEmailId, setGoalEmailId] = useState<string>("Email-4");
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SearchResponse | null>(null);
  const [activeTab, setActiveTab] = useState<"graph" | "trace" | "path">("graph");
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);

  // Load demo dataset on initial mount
  useEffect(() => {
    loadDemo();
  }, []);

  const loadDemo = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchSearchDemoEmails();
      setEmails(data.emails);
      if (data.emails.length >= 4) {
        setStartEmailId(data.emails[0].id);
        setGoalEmailId(data.emails[3].id);
      } else if (data.emails.length >= 2) {
        setStartEmailId(data.emails[0].id);
        setGoalEmailId(data.emails[1].id);
      }
      setResult(null);
    } catch (err: any) {
      setError(err.message || "Failed to load demo data.");
    } finally {
      setLoading(false);
    }
  };

  const handleAddEmail = () => {
    const nextIdx = emails.length + 1;
    const newId = `Email-${nextIdx}`;
    setEmails([
      ...emails,
      {
        id: newId,
        subject: `Investigation Message ${nextIdx}`,
        body: `Please review security update at https://portal-${nextIdx}.example.com/check`,
      },
    ]);
  };

  const handleRemoveEmail = (idToRemove: string) => {
    if (emails.length <= 2) {
      setError("At least two email records are required to perform path search.");
      return;
    }
    const filtered = emails.filter((e) => e.id !== idToRemove);
    setEmails(filtered);
    if (startEmailId === idToRemove) {
      setStartEmailId(filtered[0]?.id || "");
    }
    if (goalEmailId === idToRemove) {
      setGoalEmailId(filtered[filtered.length - 1]?.id || "");
    }
    setResult(null);
  };

  const handleUpdateEmail = (id: string, field: "id" | "subject" | "body", value: string) => {
    setEmails((prev) =>
      prev.map((e) => (e.id === id ? { ...e, [field]: value } : e))
    );
  };

  const handleCsvUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const text = e.target?.result as string;
        const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
        if (lines.length < 2) {
          setError("CSV file must contain a header row and at least one data row.");
          return;
        }

        const headers = lines[0].split(",").map((h) => h.trim().toLowerCase().replace(/['"]/g, ""));
        const idCol = headers.findIndex((h) => h === "id" || h === "message_id" || h === "email_id");
        const subjCol = headers.findIndex((h) => h === "subject" || h === "title");
        const bodyCol = headers.findIndex((h) => h === "body" || h === "text" || h === "content" || h === "message");

        if (bodyCol === -1) {
          setError("CSV must contain a 'body', 'text', or 'message' column.");
          return;
        }

        const parsed: InvestigationEmailItem[] = [];
        for (let i = 1; i < lines.length && parsed.length < 25; i++) {
          const row = lines[i].split(",");
          if (row.length <= bodyCol) continue;
          const bodyVal = row.slice(bodyCol).join(",").replace(/^["']|["']$/g, "").trim();
          const subjVal = subjCol !== -1 && row[subjCol] ? row[subjCol].replace(/^["']|["']$/g, "").trim() : `Email ${i}`;
          const idVal = idCol !== -1 && row[idCol] ? row[idCol].replace(/^["']|["']$/g, "").trim() : `Email-${i}`;

          if (bodyVal) {
            parsed.push({
              id: idVal,
              subject: subjVal,
              body: bodyVal,
            });
          }
        }

        if (parsed.length >= 2) {
          setEmails(parsed);
          setStartEmailId(parsed[0].id);
          setGoalEmailId(parsed[parsed.length - 1].id);
          setResult(null);
          setError(null);
        } else {
          setError("Could not parse at least two valid email rows from the CSV.");
        }
      } catch (err: any) {
        setError(`Failed to read CSV: ${err.message}`);
      }
    };
    reader.readAsText(file);
    // Reset file input
    event.target.value = "";
  };

  const handleRunSearch = async () => {
    setError(null);
    if (!startEmailId || !goalEmailId) {
      setError("Please select both a Starting Email and a Target Email.");
      return;
    }
    if (emails.length < 2) {
      setError("Please supply at least two emails.");
      return;
    }

    setLoading(true);
    try {
      const res = await runGreedyBestFirstSearch(emails, startEmailId, goalEmailId);
      setResult(res);
      setActiveTab("graph");
    } catch (err: any) {
      setError(err.message || "Search failed unexpectedly.");
    } finally {
      setLoading(false);
    }
  };

  // Node & Edge Coordinate Layout for SVG Visualization
  const graphLayout = useMemo(() => {
    if (!result || !result.graph_visualization) return null;

    const nodes = result.graph_visualization.nodes;
    const edges = result.graph_visualization.edges;

    const emailNodes = nodes.filter((n) => n.type === "email");
    const indicatorNodes = nodes.filter((n) => n.type === "indicator");

    const width = 840;
    const height = 480;

    const coords: Record<string, { x: number; y: number }> = {};

    // Layout Emails on Left column or outer circle
    const emailX = 140;
    const emailSpacing = height / (emailNodes.length + 1);
    emailNodes.forEach((node, idx) => {
      coords[node.id] = {
        x: emailX,
        y: emailSpacing * (idx + 1),
      };
    });

    // Layout Indicators on Right columns
    const indicatorX = 580;
    const indSpacing = height / (indicatorNodes.length + 1);
    indicatorNodes.forEach((node, idx) => {
      // Slight horizontal stagger for readability
      const xOffset = (idx % 2 === 0 ? 0 : 50);
      coords[node.id] = {
        x: indicatorX + xOffset,
        y: indSpacing * (idx + 1),
      };
    });

    return { width, height, coords, nodes, edges };
  }, [result]);

  return (
    <section id="investigation" className="relative py-24 px-4 sm:px-6 lg:px-8 border-t border-white/[0.08] bg-[#0B0B0D]">
      <div className="max-w-6xl mx-auto space-y-12">
        {/* Section Header */}
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-6">
          <div className="space-y-3">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[#B8F44A]/10 border border-[#B8F44A]/30 text-[#B8F44A] text-xs font-mono uppercase tracking-widest">
              <Compass className="w-3.5 h-3.5" />
              <span>CO2 Syllabus Alignment • Informed Search</span>
            </div>
            <h2 className="text-3xl sm:text-4xl font-sans font-bold tracking-tight text-[#F4F3F0]">
              Threat Investigation Explorer
            </h2>
            <p className="text-sm sm:text-base text-[#A3A3A8] max-w-2xl font-sans">
              Demonstrates <strong>Greedy Best-First Search</strong> on an email-indicator bipartite graph.
              The algorithm evaluates Jaccard structural similarity <span className="font-mono text-[#B8F44A]">h(n) = 1 - J(I(n), I(goal))</span> to discover shared digital infrastructure (URLs, domains, email addresses).
            </p>
          </div>

          {/* Quick Action Buttons */}
          <div className="flex items-center gap-3">
            <button
              onClick={loadDemo}
              disabled={loading}
              className="inline-flex items-center gap-2 px-3.5 py-2 rounded-xl bg-[#141418] border border-white/[0.12] hover:border-[#B8F44A]/40 text-xs font-mono uppercase text-[#F4F3F0] hover:text-[#B8F44A] transition-all disabled:opacity-50"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
              <span>Load Demo Data</span>
            </button>

            <label className="inline-flex items-center gap-2 px-3.5 py-2 rounded-xl bg-[#141418] border border-white/[0.12] hover:border-[#B8F44A]/40 text-xs font-mono uppercase text-[#F4F3F0] hover:text-[#B8F44A] cursor-pointer transition-all">
              <Upload className="w-3.5 h-3.5" />
              <span>Upload CSV</span>
              <input
                type="file"
                accept=".csv"
                onChange={handleCsvUpload}
                className="hidden"
              />
            </label>
          </div>
        </div>

        {/* Error Alert */}
        {error && (
          <div className="p-4 rounded-xl bg-[#EF4444]/10 border border-[#EF4444]/30 flex items-start gap-3 text-sm text-[#EF4444]">
            <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
            <div className="space-y-1">
              <p className="font-semibold">Investigation Alert</p>
              <p className="text-xs text-[#FCA5A5]">{error}</p>
            </div>
          </div>
        )}

        {/* Input Configuration Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Left Column: Email Document Collection (7 cols) */}
          <div className="lg:col-span-7 space-y-4">
            <div className="flex items-center justify-between pb-2 border-b border-white/[0.08]">
              <div className="flex items-center gap-2">
                <Mail className="w-4 h-4 text-[#B8F44A]" />
                <span className="font-mono text-xs uppercase tracking-wider text-[#F4F3F0] font-semibold">
                  Investigation Email Corpus ({emails.length} Records)
                </span>
              </div>
              <button
                onClick={handleAddEmail}
                className="inline-flex items-center gap-1.5 text-[11px] font-mono uppercase text-[#B8F44A] hover:underline"
              >
                <Plus className="w-3 h-3" />
                <span>Add Email</span>
              </button>
            </div>

            <div className="space-y-3 max-h-[460px] overflow-y-auto pr-2 scrollbar-thin">
              {emails.map((email, idx) => {
                const isStart = startEmailId === email.id;
                const isGoal = goalEmailId === email.id;
                return (
                  <div
                    key={email.id}
                    className={`p-3.5 rounded-xl border transition-all duration-200 ${
                      isStart
                        ? "bg-[#3B82F6]/10 border-[#3B82F6]/50 shadow-lg"
                        : isGoal
                        ? "bg-[#B8F44A]/10 border-[#B8F44A]/50 shadow-lg"
                        : "bg-[#141418] border-white/[0.08] hover:border-white/[0.16]"
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2 mb-2">
                      <div className="flex items-center gap-2">
                        <input
                          type="text"
                          value={email.id}
                          onChange={(e) => handleUpdateEmail(email.id, "id", e.target.value)}
                          className="w-24 px-2 py-0.5 rounded bg-[#0B0B0D] border border-white/[0.1] font-mono text-xs font-semibold text-[#F4F3F0] focus:border-[#B8F44A] outline-none"
                          placeholder="Email ID"
                        />
                        {isStart && (
                          <span className="px-2 py-0.5 rounded-full text-[10px] font-mono uppercase bg-[#3B82F6] text-white font-bold">
                            START NODE
                          </span>
                        )}
                        {isGoal && (
                          <span className="px-2 py-0.5 rounded-full text-[10px] font-mono uppercase bg-[#B8F44A] text-black font-bold">
                            TARGET GOAL
                          </span>
                        )}
                      </div>

                      {emails.length > 2 && (
                        <button
                          onClick={() => handleRemoveEmail(email.id)}
                          className="p-1 rounded text-[#A3A3A8] hover:text-[#EF4444] transition-colors"
                          title="Remove email"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      )}
                    </div>

                    <div className="space-y-1.5">
                      <input
                        type="text"
                        value={email.subject}
                        onChange={(e) => handleUpdateEmail(email.id, "subject", e.target.value)}
                        className="w-full px-2.5 py-1 rounded bg-[#0B0B0D]/80 border border-white/[0.08] text-xs text-[#F4F3F0] focus:border-[#B8F44A] outline-none placeholder:text-[#555]"
                        placeholder="Subject Line"
                      />
                      <textarea
                        value={email.body}
                        onChange={(e) => handleUpdateEmail(email.id, "body", e.target.value)}
                        rows={2}
                        className="w-full px-2.5 py-1.5 rounded bg-[#0B0B0D]/80 border border-white/[0.08] text-xs font-mono text-[#D4D4D8] focus:border-[#B8F44A] outline-none placeholder:text-[#555] resize-none"
                        placeholder="Email body text with URLs or emails..."
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Right Column: Search Execution & Controls (5 cols) */}
          <div className="lg:col-span-5 space-y-6">
            <div className="p-5 rounded-2xl bg-[#141418] border border-white/[0.08] space-y-5">
              <div className="flex items-center gap-2 pb-2 border-b border-white/[0.08]">
                <Compass className="w-4 h-4 text-[#B8F44A]" />
                <span className="font-mono text-xs uppercase tracking-wider text-[#F4F3F0] font-semibold">
                  Search Traversal Setup
                </span>
              </div>

              {/* Start Node Selector */}
              <div className="space-y-1.5">
                <label className="text-xs font-mono uppercase text-[#A3A3A8] flex items-center justify-between">
                  <span>1. Starting Email (Root):</span>
                  <span className="text-[#3B82F6] font-semibold">Priority h(n)</span>
                </label>
                <select
                  value={startEmailId}
                  onChange={(e) => setStartEmailId(e.target.value)}
                  className="w-full px-3 py-2 rounded-xl bg-[#0B0B0D] border border-white/[0.12] text-xs font-mono text-[#F4F3F0] focus:border-[#3B82F6] outline-none"
                >
                  {emails.map((e) => (
                    <option key={`start-${e.id}`} value={e.id}>
                      {e.id}: {e.subject ? e.subject.slice(0, 35) : "No Subject"}
                    </option>
                  ))}
                </select>
              </div>

              {/* Target Goal Selector */}
              <div className="space-y-1.5">
                <label className="text-xs font-mono uppercase text-[#A3A3A8] flex items-center justify-between">
                  <span>2. Target Email (Goal):</span>
                  <span className="text-[#B8F44A] font-semibold">h(goal) = 0.0</span>
                </label>
                <select
                  value={goalEmailId}
                  onChange={(e) => setGoalEmailId(e.target.value)}
                  className="w-full px-3 py-2 rounded-xl bg-[#0B0B0D] border border-white/[0.12] text-xs font-mono text-[#F4F3F0] focus:border-[#B8F44A] outline-none"
                >
                  {emails.map((e) => (
                    <option key={`goal-${e.id}`} value={e.id}>
                      {e.id}: {e.subject ? e.subject.slice(0, 35) : "No Subject"}
                    </option>
                  ))}
                </select>
              </div>

              {/* Traversal Button */}
              <button
                onClick={handleRunSearch}
                disabled={loading || emails.length < 2}
                className="w-full py-3 px-4 rounded-xl bg-[#B8F44A] hover:bg-[#a3df3d] text-black font-mono text-xs uppercase font-bold tracking-wider flex items-center justify-center gap-2 transition-all shadow-lg hover:shadow-[#B8F44A]/20 disabled:opacity-50"
              >
                {loading ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    <span>Executing Greedy Search...</span>
                  </>
                ) : (
                  <>
                    <Search className="w-4 h-4" />
                    <span>Run Greedy Best-First Search</span>
                  </>
                )}
              </button>

              {/* Heuristic Formula Card */}
              <div className="p-3.5 rounded-xl bg-[#0B0B0D] border border-white/[0.06] space-y-2 text-xs">
                <div className="flex items-center gap-2 text-[#B8F44A] font-mono text-[11px] font-semibold uppercase">
                  <Info className="w-3.5 h-3.5" />
                  <span>Heuristic Definition</span>
                </div>
                <p className="text-[#A3A3A8] leading-relaxed text-[11px]">
                  Priority is guided strictly by structural dissimilarity to the target indicator set <span className="text-white font-mono">I(goal)</span>:
                </p>
                <div className="font-mono text-[11px] bg-[#141418] p-2 rounded text-[#F4F3F0] text-center border border-white/[0.08]">
                  h(n) = 1.0 - (|I(n) ∩ I(goal)| / |I(n) ∪ I(goal)|)
                </div>
                <p className="text-[10px] text-[#71717A] italic">
                  Note: Greedy BFS expands lowest h(n) without considering step cost g(n).
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Results Panel */}
        {result && (
          <div className="space-y-6 pt-6 border-t border-white/[0.08] animate-in fade-in duration-300">
            {/* Top Summary Banner */}
            <div
              className={`p-5 rounded-2xl border flex flex-col md:flex-row md:items-center justify-between gap-4 ${
                result.found
                  ? "bg-[#54D6A0]/10 border-[#54D6A0]/40"
                  : "bg-[#F59E0B]/10 border-[#F59E0B]/40"
              }`}
            >
              <div className="flex items-start gap-3">
                {result.found ? (
                  <CheckCircle2 className="w-6 h-6 text-[#54D6A0] flex-shrink-0 mt-0.5" />
                ) : (
                  <AlertCircle className="w-6 h-6 text-[#F59E0B] flex-shrink-0 mt-0.5" />
                )}
                <div>
                  <h3 className="text-base font-bold text-[#F4F3F0]">
                    {result.found
                      ? `Target Reached: Connecting Path Discovered (${result.search_metrics.path_length} hops)`
                      : "Search Terminated: Disconnected Components"}
                  </h3>
                  <p className="text-xs text-[#A3A3A8] mt-0.5">{result.message}</p>
                </div>
              </div>

              {/* Stat Badges */}
              <div className="flex flex-wrap items-center gap-2">
                <div className="px-3 py-1.5 rounded-lg bg-[#0B0B0D] border border-white/[0.08] font-mono text-xs">
                  <span className="text-[#A3A3A8]">Nodes Explored: </span>
                  <span className="text-[#B8F44A] font-bold">{result.search_metrics.nodes_explored}</span>
                </div>
                <div className="px-3 py-1.5 rounded-lg bg-[#0B0B0D] border border-white/[0.08] font-mono text-xs">
                  <span className="text-[#A3A3A8]">Graph Size: </span>
                  <span className="text-white font-bold">{result.graph_summary.total_nodes} nodes</span>
                </div>
                <div className="px-3 py-1.5 rounded-lg bg-[#0B0B0D] border border-white/[0.08] font-mono text-xs">
                  <span className="text-[#A3A3A8]">Max Frontier: </span>
                  <span className="text-[#38BDF8] font-bold">{result.search_metrics.max_frontier_size}</span>
                </div>
              </div>
            </div>

            {/* Path Breadcrumbs (if found) */}
            {result.found && result.path.length > 0 && (
              <div className="p-4 rounded-xl bg-[#141418] border border-white/[0.08] space-y-3">
                <span className="font-mono text-xs uppercase tracking-wider text-[#A3A3A8]">
                  Identified Traversal Sequence ({result.path.length} vertices)
                </span>
                <div className="flex flex-wrap items-center gap-2">
                  {result.path.map((node, idx) => {
                    const isEmail = node.node_type === "email";
                    const isLast = idx === result.path.length - 1;
                    return (
                      <React.Fragment key={`path-pill-${node.node_id}`}>
                        <div
                          className={`px-3 py-1.5 rounded-xl border flex items-center gap-2 transition-all ${
                            isEmail
                              ? "bg-[#3B82F6]/15 border-[#3B82F6]/40 text-[#60A5FA]"
                              : "bg-[#B8F44A]/15 border-[#B8F44A]/40 text-[#B8F44A]"
                          }`}
                        >
                          {isEmail ? <Mail className="w-3.5 h-3.5" /> : <Globe className="w-3.5 h-3.5" />}
                          <span className="font-mono text-xs font-semibold">{node.display_label}</span>
                          <span className="px-1.5 py-0.2 rounded text-[10px] font-mono bg-black/40 text-white/80">
                            h={node.heuristic}
                          </span>
                        </div>
                        {!isLast && <ArrowRight className="w-3.5 h-3.5 text-[#555]" />}
                      </React.Fragment>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Visualization and Trace Tabs */}
            <div className="space-y-4">
              <div className="flex items-center gap-2 border-b border-white/[0.08]">
                <button
                  onClick={() => setActiveTab("graph")}
                  className={`pb-3 px-3 font-mono text-xs uppercase tracking-wider transition-colors border-b-2 flex items-center gap-2 ${
                    activeTab === "graph"
                      ? "border-[#B8F44A] text-[#B8F44A] font-bold"
                      : "border-transparent text-[#A3A3A8] hover:text-[#F4F3F0]"
                  }`}
                >
                  <Network className="w-3.5 h-3.5" />
                  <span>Bipartite Graph Map</span>
                </button>
                <button
                  onClick={() => setActiveTab("trace")}
                  className={`pb-3 px-3 font-mono text-xs uppercase tracking-wider transition-colors border-b-2 flex items-center gap-2 ${
                    activeTab === "trace"
                      ? "border-[#B8F44A] text-[#B8F44A] font-bold"
                      : "border-transparent text-[#A3A3A8] hover:text-[#F4F3F0]"
                  }`}
                >
                  <Layers className="w-3.5 h-3.5" />
                  <span>Frontier Exploration Trace</span>
                </button>
              </div>

              {/* Tab 1: Interactive SVG Graph */}
              {activeTab === "graph" && graphLayout && (
                <div className="p-4 rounded-2xl bg-[#141418] border border-white/[0.08] space-y-3">
                  <div className="flex items-center justify-between text-xs font-mono text-[#A3A3A8]">
                    <span>Left: Email Nodes • Right: Extracted Digital Indicators</span>
                    <span className="text-[#B8F44A]">● Neon Green = Discovered Path</span>
                  </div>

                  <div className="relative overflow-x-auto rounded-xl bg-[#0B0B0D] border border-white/[0.06] p-2">
                    <svg
                      viewBox={`0 0 ${graphLayout.width} ${graphLayout.height}`}
                      className="w-full min-w-[700px] h-[480px]"
                    >
                      {/* Edges */}
                      {graphLayout.edges.map((edge, idx) => {
                        const sourcePos = graphLayout.coords[edge.source];
                        const targetPos = graphLayout.coords[edge.target];
                        if (!sourcePos || !targetPos) return null;

                        return (
                          <line
                            key={`edge-${idx}`}
                            x1={sourcePos.x}
                            y1={sourcePos.y}
                            x2={targetPos.x}
                            y2={targetPos.y}
                            stroke={edge.is_path ? "#B8F44A" : "rgba(255, 255, 255, 0.08)"}
                            strokeWidth={edge.is_path ? 2.5 : 1}
                            strokeDasharray={edge.is_path ? "none" : "3,3"}
                          />
                        );
                      })}

                      {/* Nodes */}
                      {graphLayout.nodes.map((node) => {
                        const pos = graphLayout.coords[node.id];
                        if (!pos) return null;

                        const isEmail = node.type === "email";
                        const isPath = node.is_path;
                        const isHovered = hoveredNode === node.id;

                        return (
                          <g
                            key={`node-${node.id}`}
                            transform={`translate(${pos.x}, ${pos.y})`}
                            onMouseEnter={() => setHoveredNode(node.id)}
                            onMouseLeave={() => setHoveredNode(null)}
                            className="cursor-pointer transition-transform"
                          >
                            {isEmail ? (
                              // Email Node (Rect)
                              <>
                                <rect
                                  x="-55"
                                  y="-18"
                                  width="110"
                                  height="36"
                                  rx="8"
                                  fill={isPath ? "rgba(59, 130, 246, 0.3)" : "#141418"}
                                  stroke={
                                    isPath
                                      ? "#3B82F6"
                                      : isHovered
                                      ? "#B8F44A"
                                      : "rgba(255, 255, 255, 0.2)"
                                  }
                                  strokeWidth={isPath ? 2 : 1}
                                />
                                <text
                                  textAnchor="middle"
                                  y="4"
                                  fill="#F4F3F0"
                                  fontSize="10"
                                  fontFamily="monospace"
                                  fontWeight="bold"
                                >
                                  {node.label.slice(0, 14)}
                                </text>
                              </>
                            ) : (
                              // Indicator Node (Circle / Pill)
                              <>
                                <circle
                                  r={isPath ? 12 : 9}
                                  fill={isPath ? "#B8F44A" : "#1A1A20"}
                                  stroke={isPath ? "#B8F44A" : "rgba(255, 255, 255, 0.2)"}
                                  strokeWidth={isPath ? 2 : 1}
                                />
                                <text
                                  x="16"
                                  y="4"
                                  fill={isPath ? "#B8F44A" : "#A3A3A8"}
                                  fontSize="10"
                                  fontFamily="monospace"
                                >
                                  {node.label.length > 25 ? `${node.label.slice(0, 25)}...` : node.label}
                                </text>
                              </>
                            )}
                          </g>
                        );
                      })}
                    </svg>

                    {/* Tooltip Overlay */}
                    {hoveredNode && (
                      <div className="absolute bottom-3 left-3 p-2.5 rounded-lg bg-[#141418]/95 border border-[#B8F44A]/40 text-xs font-mono text-[#F4F3F0] shadow-xl pointer-events-none">
                        <span className="text-[#B8F44A]">Selected: </span>
                        {graphLayout.nodes.find((n) => n.id === hoveredNode)?.label}
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Tab 2: Frontier Exploration Trace Table */}
              {activeTab === "trace" && (
                <div className="overflow-x-auto rounded-xl bg-[#141418] border border-white/[0.08]">
                  <table className="w-full text-left text-xs font-mono">
                    <thead className="bg-[#0B0B0D] text-[#A3A3A8] uppercase tracking-wider border-b border-white/[0.08]">
                      <tr>
                        <th className="py-2.5 px-3">Order</th>
                        <th className="py-2.5 px-3">Node Type</th>
                        <th className="py-2.5 px-3">Identifier / Label</th>
                        <th className="py-2.5 px-3">Heuristic h(n)</th>
                        <th className="py-2.5 px-3">Evaluation Rationale</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-white/[0.04]">
                      {result.explored.map((item, idx) => {
                        const isEmail = item.node_type === "email";
                        return (
                          <tr
                            key={`trace-${item.node_id}-${idx}`}
                            className="hover:bg-white/[0.02] transition-colors"
                          >
                            <td className="py-2.5 px-3 text-[#B8F44A] font-bold">#{item.step_order}</td>
                            <td className="py-2.5 px-3">
                              <span
                                className={`px-2 py-0.5 rounded text-[10px] ${
                                  isEmail
                                    ? "bg-[#3B82F6]/20 text-[#60A5FA]"
                                    : "bg-[#54D6A0]/20 text-[#54D6A0]"
                                }`}
                              >
                                {item.node_type.toUpperCase()}
                              </span>
                            </td>
                            <td className="py-2.5 px-3 text-[#F4F3F0] font-semibold">{item.display_label}</td>
                            <td className="py-2.5 px-3">
                              <span
                                className={`font-bold ${
                                  item.heuristic === 0
                                    ? "text-[#54D6A0]"
                                    : item.heuristic < 1
                                    ? "text-[#B8F44A]"
                                    : "text-[#A3A3A8]"
                                }`}
                              >
                                {item.heuristic}
                              </span>
                            </td>
                            <td className="py-2.5 px-3 text-[#A3A3A8]">{item.explanation}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </div>

            {/* Educational Theory Section */}
            <div className="p-5 rounded-2xl bg-[#141418] border border-white/[0.08] space-y-3">
              <div className="flex items-center gap-2 text-[#B8F44A] font-mono text-xs font-semibold uppercase">
                <HelpCircle className="w-4 h-4" />
                <span>Academic Theory: How Greedy Best-First Search Operates</span>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs text-[#A3A3A8] pt-1">
                <div className="p-3 rounded-xl bg-[#0B0B0D] border border-white/[0.06] space-y-1.5">
                  <span className="text-[#F4F3F0] font-semibold block">1. Priority Queue Ordering</span>
                  <p>
                    Frontier states are stored in a min-heap prioritizing solely by <span className="text-[#B8F44A] font-mono">Priority(n) = h(n)</span>.
                    Unlike A*, Greedy Best-First Search disregards past path cost <span className="font-mono">g(n)</span> to aggressively seek proximity to the goal.
                  </p>
                </div>
                <div className="p-3 rounded-xl bg-[#0B0B0D] border border-white/[0.06] space-y-1.5">
                  <span className="text-[#F4F3F0] font-semibold block">2. Jaccard Structural Heuristic</span>
                  <p>
                    The heuristic calculates set dissimilarity between indicator sets: <span className="font-mono text-white">1 - J(I(n), I(goal))</span>.
                    Nodes sharing identical infrastructure with the target receive <span className="text-[#54D6A0] font-mono">h = 0.0</span> and are expanded first.
                  </p>
                </div>
                <div className="p-3 rounded-xl bg-[#0B0B0D] border border-white/[0.06] space-y-1.5">
                  <span className="text-[#F4F3F0] font-semibold block">3. Cybersecurity Context</span>
                  <p>
                    A connecting path signifies shared digital artifacts (phishing domains, hostnames, or sender addresses).
                    This search demonstrates graph-based investigation; our Naïve Bayes model continues to perform statistical Spam/Ham classification.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}

