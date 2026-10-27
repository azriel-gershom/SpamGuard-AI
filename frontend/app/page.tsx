import React from "react";
import Navbar from "@/components/Navbar";
import Hero from "@/components/Hero";
import ProblemSection from "@/components/ProblemSection";
import PipelineSection from "@/components/PipelineSection";
import EmailScanner from "@/components/EmailScanner";
import BatchScanner from "@/components/BatchScanner";
import ThreatInvestigationExplorer from "@/components/ThreatInvestigationExplorer";
import ModelAnalytics from "@/components/ModelAnalytics";
import Methodology from "@/components/Methodology";
import Footer from "@/components/Footer";

export default function Home() {
  return (
    <main className="min-h-screen flex flex-col bg-[#0B0B0D] text-[#F4F3F0]">
      <Navbar />
      <Hero />
      <ProblemSection />
      <PipelineSection />
      <EmailScanner />
      <BatchScanner />
      <ThreatInvestigationExplorer />
      <ModelAnalytics />
      <Methodology />
      <Footer />
    </main>
  );
}
