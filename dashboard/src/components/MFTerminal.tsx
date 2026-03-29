"use client";

import React, { useState, useEffect } from "react";
import { 
  ShieldCheck, 
  TrendingUp, 
  AlertCircle, 
  Loader2,
  Table as TableIcon,
  ChevronRight,
  Zap,
  Target
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

export const MFTerminal = ({ config }: { config?: any }) => {
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);

  const fetchAnalysis = async () => {
    setLoading(true);
    setError(null);
    try {
      const resp = await fetch("/api/mf-analysis");
      if (!resp.ok) throw new Error("Analysis failed");
      const data = await resp.json();
      setResults(data);
    } catch (err: any) {
      setError(err.message || "An unexpected error occurred");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAnalysis();
  }, []);

  return (
    <div className="space-y-8">
      {/* Header Metric Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-zinc-900/40 border border-zinc-800/80 p-6 rounded-3xl">
          <p className="text-zinc-500 text-xs font-bold uppercase tracking-widest mb-1">Consistency Score Average</p>
          <div className="flex items-center gap-2">
            <h3 className="text-3xl font-black text-white">88.4</h3>
            <span className="text-emerald-500 font-bold text-sm">↑ 2.4%</span>
          </div>
        </div>
        <div className="bg-zinc-900/40 border border-zinc-800/80 p-6 rounded-3xl">
          <p className="text-zinc-500 text-xs font-bold uppercase tracking-widest mb-1">Benchmark Alpha</p>
          <div className="flex items-center gap-2">
            <h3 className="text-3xl font-black text-white">12.8%</h3>
            <span className="text-emerald-500 font-bold text-sm">Outperforming</span>
          </div>
        </div>
        <div className="bg-zinc-900/40 border border-zinc-800/80 p-6 rounded-3xl relative overflow-hidden group">
          <p className="text-zinc-500 text-xs font-bold uppercase tracking-widest mb-1">Active Monitoring</p>
          <div className="flex items-center gap-2">
            <h3 className="text-3xl font-black text-white">42 Funds</h3>
            <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />
          </div>
          <div className="absolute right-0 bottom-0 opacity-10 group-hover:scale-110 transition-transform">
            <Zap className="w-24 h-24 text-blue-500" />
          </div>
        </div>
      </div>

      {/* Main Analysis Table */}
      <div className="bg-zinc-900/30 border border-zinc-800/50 rounded-[32px] overflow-hidden backdrop-blur-md">
        <div className="p-8 border-b border-zinc-800/60 flex items-center justify-between">
          <h3 className="text-xl font-black flex items-center gap-3 text-white">
            <ShieldCheck className="text-emerald-500" />
            MF Consistency Rankings
          </h3>
          <button 
            onClick={fetchAnalysis}
            disabled={loading}
            className="text-xs font-black uppercase tracking-widest text-blue-500 hover:text-blue-400 transition-colors flex items-center gap-2"
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <TrendingUp className="w-4 h-4" />}
            Refresh Data
          </button>
        </div>

        <div className="min-h-[400px]">
          <AnimatePresence mode="wait">
            {loading ? (
              <motion.div 
                key="loading"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="p-32 flex flex-col items-center justify-center space-y-4"
              >
                <div className="w-12 h-12 border-4 border-emerald-500/20 border-t-emerald-500 rounded-full animate-spin" />
                <p className="text-zinc-500 font-bold tracking-tight text-sm uppercase">Recalculating Rolling Alpha...</p>
              </motion.div>
            ) : error ? (
              <motion.div 
                key="error"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="p-32 flex flex-col items-center justify-center text-center space-y-4"
              >
                <div className="p-4 bg-rose-500/10 text-rose-500 rounded-full">
                  <AlertCircle className="w-12 h-12" />
                </div>
                <h3 className="text-xl font-bold text-white">Benchmarking Timeout</h3>
                <p className="text-zinc-500 max-w-md italic">{error}</p>
              </motion.div>
            ) : (
              <motion.div 
                key="table"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="overflow-x-auto"
              >
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="bg-zinc-950/40">
                      <th className="px-8 py-4 text-xs font-black uppercase tracking-widest text-zinc-600">Scheme Name / Code</th>
                      <th className="px-8 py-4 text-xs font-black uppercase tracking-widest text-zinc-600">Category</th>
                      <th className="px-8 py-4 text-xs font-black uppercase tracking-widest text-zinc-600">Alpha</th>
                      <th className="px-8 py-4 text-xs font-black uppercase tracking-widest text-zinc-600">Consistency</th>
                      <th className="px-8 py-4 text-xs font-black uppercase tracking-widest text-zinc-600">Integrity</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-zinc-800/40">
                    {results.map((fund, i) => (
                      <tr key={i} className="hover:bg-zinc-900/40 transition-colors group">
                        <td className="px-8 py-6">
                          <div className="flex items-center gap-4">
                            <div className="w-10 h-10 bg-zinc-900 rounded-xl flex items-center justify-center font-bold text-xs text-zinc-400 group-hover:bg-emerald-600 group-hover:text-white transition-all shadow-sm">
                              {String(fund["Scheme Code"] || "MF").substring(0, 2).toUpperCase()}
                            </div>
                            <div>
                                <p className="font-bold text-white leading-tight">{fund["Scheme"] || "Growth Fund"}</p>
                                <p className="text-[10px] font-bold text-zinc-600 uppercase tracking-widest mt-0.5">Rolling Returns (3Y)</p>
                            </div>
                          </div>
                        </td>
                        <td className="px-8 py-6">
                          <span className="px-3 py-1 bg-zinc-800 rounded-lg text-xs font-bold text-zinc-300">
                            {fund["Category"] || "Equity"}
                          </span>
                        </td>
                        <td className="px-8 py-6">
                            <p className="font-black text-emerald-400">+{parseFloat(fund["Alpha"] || "0").toFixed(2)}%</p>
                        </td>
                        <td className="px-8 py-6">
                            <div className="w-full max-w-[120px] bg-zinc-800 h-2 rounded-full overflow-hidden">
                                <motion.div 
                                    initial={{ width: 0 }}
                                    animate={{ width: `${fund["Conviction Score"] || fund["Consistency Score"] || 0}%` }}
                                    className="h-full bg-blue-500 rounded-full"
                                />
                            </div>
                            <p className="text-[10px] font-black text-zinc-600 mt-1 uppercase tracking-widest">
                                {Math.round(fund["Conviction Score"] || fund["Consistency Score"] || 0)}% Consistency
                            </p>
                        </td>
                        <td className="px-8 py-6">
                            <div className="flex items-center gap-2">
                                <div className="w-2 h-2 rounded-full bg-emerald-500" />
                                <span className="text-xs font-black uppercase tracking-tighter text-emerald-500">Verified</span>
                            </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
};
