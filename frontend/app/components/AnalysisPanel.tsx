"use client";

import { useState } from "react";
import {
  BarChart2, BookOpen, Tag, Loader2,
  ChevronDown, ChevronUp, AlertCircle,
  Sparkles, Clock,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { analyzeDocument } from "@/lib/api";
import { getCategoryColor } from "@/lib/utils";
import type { FullAnalysisResponse } from "@/types";

interface AnalysisPanelProps {
  documentId: string;
  filename: string;
}

export default function AnalysisPanel({
  documentId,
  filename,
}: AnalysisPanelProps) {
  const [analysis, setAnalysis]         = useState<FullAnalysisResponse | null>(null);
  const [loading, setLoading]           = useState(false);
  const [error, setError]               = useState("");
  const [showAllScores, setShowAllScores] = useState(false);
  const [maxWords, setMaxWords]         = useState(400);

  const handleAnalyze = async () => {
    setLoading(true);
    setError("");
    try {
      const res = await analyzeDocument(documentId, maxWords);
      setAnalysis(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Analysis failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card className="animate-fade-in">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2">
          <BarChart2 className="w-5 h-5 text-blue-600" />
          Document Analysis
        </CardTitle>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Controls */}
        {!analysis && (
          <div className="space-y-3">
            <div>
              <label className="text-xs text-gray-500 font-medium">
                Summary length (words)
              </label>
              <div className="flex items-center gap-3 mt-1">
                <input
                  type="range"
                  min={100}
                  max={1000}
                  step={100}
                  value={maxWords}
                  onChange={(e) => setMaxWords(Number(e.target.value))}
                  className="flex-1 accent-blue-600"
                />
                <span className="text-sm font-medium text-gray-700 w-12 text-right">
                  ~{maxWords}w
                </span>
              </div>
            </div>

            <Button
              onClick={handleAnalyze}
              disabled={loading}
              className="w-full"
            >
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Analyzing — this may take 10–30s ...
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  Run Full Analysis
                </>
              )}
            </Button>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="flex items-center gap-2 p-3 bg-red-50
                          border border-red-200 rounded-lg
                          text-sm text-red-700 animate-fade-in">
            <AlertCircle className="w-4 h-4 flex-shrink-0" />
            {error}
          </div>
        )}

        {/* Results */}
        {analysis && (
          <div className="space-y-5 animate-slide-up">

            {/* Timing */}
            <div className="flex items-center gap-1.5 text-xs text-gray-400">
              <Clock className="w-3.5 h-3.5" />
              Completed in {analysis.total_processing_time.toFixed(2)}s
            </div>

            {/* ── Classification ─────────────────────────────────────────── */}
            {analysis.classification && !("error" in analysis.classification) && (
              <div className="space-y-3">
                <h3 className="text-sm font-semibold text-gray-700
                               flex items-center gap-2">
                  <Tag className="w-4 h-4 text-purple-500" />
                  Classification
                </h3>

                {/* Predicted category */}
                <div className="flex items-center justify-between">
                  <span
                    className={`inline-flex items-center px-3 py-1 rounded-full
                                text-sm font-medium border
                                ${getCategoryColor(
                                  analysis.classification.predicted_category
                                )}`}
                  >
                    {analysis.classification.predicted_category}
                  </span>
                  <Badge variant="success">
                    {Math.round(analysis.classification.confidence * 100)}% confident
                  </Badge>
                </div>

                {/* Reasoning */}
                {analysis.classification.reasoning && (
                  <p className="text-xs text-gray-500 italic leading-relaxed bg-gray-50
                                px-3 py-2 rounded-lg border border-gray-100">
                    "{analysis.classification.reasoning}"
                  </p>
                )}

                {/* Score bars */}
                <div>
                  <button
                    onClick={() => setShowAllScores((s) => !s)}
                    className="flex items-center gap-1 text-xs text-gray-400
                               hover:text-blue-600 transition-colors mb-2"
                  >
                    {showAllScores
                      ? <ChevronUp className="w-3 h-3" />
                      : <ChevronDown className="w-3 h-3" />
                    }
                    {showAllScores ? "Hide" : "Show"} all category scores
                  </button>

                  {showAllScores && (
                    <div className="space-y-2 animate-fade-in">
                      {Object.entries(analysis.classification.all_scores)
                        .sort(([, a], [, b]) => b - a)
                        .map(([cat, score]) => {
                          const isPredicted =
                            cat === analysis.classification.predicted_category;
                          return (
                            <div key={cat} className="space-y-0.5">
                              <div className="flex justify-between text-xs">
                                <span
                                  className={
                                    isPredicted
                                      ? "font-semibold text-gray-800"
                                      : "text-gray-500"
                                  }
                                >
                                  {cat}
                                </span>
                                <span className="text-gray-400">
                                  {Math.round(score * 100)}%
                                </span>
                              </div>
                              <Progress
                                value={Math.round(score * 100)}
                                className={isPredicted ? "h-2" : "h-1.5"}
                              />
                            </div>
                          );
                        })}
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Divider */}
            <hr className="border-gray-100" />

            {/* ── Summary ────────────────────────────────────────────────── */}
            {analysis.summary && !("error" in analysis.summary) && (
              <div className="space-y-2">
                <h3 className="text-sm font-semibold text-gray-700
                               flex items-center gap-2">
                  <BookOpen className="w-4 h-4 text-blue-500" />
                  Summary
                  <Badge variant="info" className="font-normal">
                    {analysis.summary.strategy}
                  </Badge>
                </h3>

                <div className="prose-chat text-sm text-gray-600
                                leading-relaxed bg-gray-50 p-4
                                rounded-lg border border-gray-100
                                max-h-72 overflow-y-auto custom-scroll">
                  {analysis.summary.text
                    .split("\n")
                    .map((line, i) => {
                      // Render bold **text** markers
                      if (line.startsWith("**") && line.endsWith("**")) {
                        return (
                          <p key={i} className="font-semibold text-gray-800 mt-3 mb-1">
                            {line.replace(/\*\*/g, "")}
                          </p>
                        );
                      }
                      if (line.startsWith("- ")) {
                        return (
                          <li key={i} className="ml-4 list-disc text-gray-600">
                            {line.slice(2)}
                          </li>
                        );
                      }
                      if (line.trim() === "") return <br key={i} />;
                      return <p key={i} className="mb-1">{line}</p>;
                    })}
                </div>

                <p className="text-xs text-gray-400">
                  Used {analysis.summary.num_chunks_used} chunk
                  {analysis.summary.num_chunks_used !== 1 ? "s" : ""} ·{" "}
                  {analysis.summary.processing_time.toFixed(2)}s
                </p>
              </div>
            )}

            {/* Re-analyze button */}
            <Button
              variant="outline"
              size="sm"
              className="w-full"
              onClick={() => {
                setAnalysis(null);
                setError("");
              }}
            >
              Re-analyze with different settings
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}