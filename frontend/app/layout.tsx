import type { Metadata } from "next";
import { Instrument_Serif, Figtree, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const instrumentSerif = Instrument_Serif({
  subsets: ["latin"],
  weight: ["400"],
  style: ["normal", "italic"],
  variable: "--font-title",
  display: "swap",
});

const figtree = Figtree({
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700"],
  variable: "--font-sans",
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "SpamGuard AI — High-Precision Email Threat Intelligence",
  description:
    "Anthracite-inspired AI spam email classification interface. 97.30% verified accuracy powered by Multinomial Naïve Bayes and explainable TF-IDF feature log-odds.",
  keywords: [
    "SpamGuard AI",
    "Spam Email Classifier",
    "Machine Learning",
    "TF-IDF",
    "Naïve Bayes",
    "Cybersecurity",
    "Phishing Detection",
  ],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="en"
      className={`${instrumentSerif.variable} ${figtree.variable} ${jetbrainsMono.variable} dark`}
    >
      <body className="bg-[#0B0B0D] text-[#F4F3F0] font-sans selection:bg-[#B8F44A] selection:text-[#0B0B0D] antialiased min-h-screen relative overflow-x-hidden">
        <div className="fixed inset-0 bg-grain pointer-events-none z-0 opacity-40" />
        <div className="relative z-10 flex flex-col min-h-screen">
          {children}
        </div>
      </body>
    </html>
  );
}
