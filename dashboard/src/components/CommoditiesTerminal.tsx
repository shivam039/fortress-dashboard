"use client";

import React, { useState, useEffect } from "react";
import { 
  Globe, 
  TrendingUp, 
  AlertCircle, 
  Loader2,
  Table as TableIcon,
  RefreshCcw,
  Zap,
  ArrowRightLeft,
  ChevronRight
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

export const CommoditiesTerminal = ({ config }: { config?: any }) => {
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);

  const fetchCommodities = async () => {
    setLoading(true);
    setError(null);
    try {
      const resp = await fetch("/api/commodities");
      if (!resp.ok) throw new Error("Commodities fetch failed");
      const data = await resp.json();
      setResults(data);
    } catch (err: any) {
      setError(err.message || "An unexpected error occurred");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchCommodities();
  }, []);

  return (
    <div className="space-y-8">
      {/* Parity Overview */}
      <div className="bg-zinc-900/40 border border-zinc-800/80 p-8 rounded-[32px] backdrop-blur-xl">
        <div className="flex items-center justify-between mb-8">
            <div className="flex items-center gap-4">
                <div className="w-12 h-12 bg-blue-600/20 text-blue-500 rounded-2xl flex items-center justify-center">
                    <Globe className="w-6 h-6" />
                </div>
                <div>
                    <h3 className="text-xl font-black text-white">Global-Local Price Parity</h3>
                    <p className="text-zinc-500 text-xs font-bold uppercase tracking-widest">Arbitrage Yield Analysis</p>
                </div>
            </div>
            <button 
                onClick={fetchCommodities}
                disabled={loading}
                className="p-3 bg-zinc-950 border border-zinc-800 rounded-2xl text-zinc-400 hover:text-white transition-all disabled:opacity-50"
            >
                {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : <RefreshCcw className="w-5 h-5" />}
            </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {loading ? (
                Array(4).fill(0).map((_, i) => (
                    <div key={i} className="bg-zinc-950/50 h-32 rounded-3xl animate-pulse border border-zinc-900" />
                ))
            ) : results.map((item, i) => (
                <motion.div 
                    key={i}
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="bg-zinc-950/50 border border-zinc-800/50 p-6 rounded-3xl hover:border-blue-500/30 transition-all group"
                >
                    <div className="flex items-center justify-between mb-3">
                        <span className="text-[10px] font-black uppercase tracking-tighter text-zinc-600">Asset</span>
                        <Zap className={`w-4 h-4 ${parseFloat(item.arb_yield) > 1.0 ? "text-emerald-500 animate-pulse" : "text-zinc-800"}`} />
                    </div>
                    <h4 className="text-2xl font-black text-white group-hover:text-blue-400 transition-colors uppercase">{item.symbol || "Asset"}</h4>
                    <div className="mt-2 flex items-center justify-between">
                        <p className="text-xs font-bold text-zinc-500">Yield</p>
                        <p className={`font-black tracking-tight ${parseFloat(item.arb_yield) > 1.0 ? "text-emerald-400" : "text-zinc-400"}`}>
                            {item.arb_yield || "0.0"}%
                        </p>
                    </div>
                </motion.div>
            ))}
        </div>
      </div>

      {/* Comparison Grid */}
      <div className="bg-zinc-900/20 border border-zinc-800/20 rounded-[40px] p-8">
        <h3 className="text-lg font-black text-zinc-400 mb-6 flex items-center gap-3">
            <ArrowRightLeft className="w-5 h-5" />
            Exchange Parity Details
        </h3>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {results.map((item, i) => (
                <div key={i} className="bg-zinc-900/40 border border-zinc-800/60 p-8 rounded-[32px] hover:border-zinc-500/20 transition-all">
                    <div className="flex items-center justify-between mb-8">
                        <div className="space-y-1">
                            <p className="text-2xl font-black text-white">{item.symbol || "Commodity"}</p>
                            <p className="text-[10px] font-black text-blue-500 uppercase tracking-widest px-2 py-0.5 bg-blue-500/10 rounded-full inline-block">Institutional Feed Active</p>
                        </div>
                        <div className="text-right">
                            <p className="text-xs font-bold text-zinc-500 uppercase">Spread</p>
                            <p className="text-lg font-black text-emerald-400">{item.spread || "0.0"}</p>
                        </div>
                    </div>

                    <div className="grid grid-cols-3 gap-8">
                        <div className="space-y-2">
                            <p className="text-[10px] font-black uppercase tracking-widest text-zinc-600">Global Price</p>
                            <p className="text-xl font-black text-white">${item.global_price || "0.0"}</p>
                        </div>
                        <div className="space-y-2 relative">
                            <div className="absolute -left-4 top-1/2 -translate-y-1/2 w-[1px] h-10 bg-zinc-800" />
                            <p className="text-[10px] font-black uppercase tracking-widest text-zinc-600">USD/INR</p>
                            <p className="text-xl font-black text-white">₹{item.usd_inr || "84.0"}</p>
                        </div>
                        <div className="space-y-2 relative">
                            <div className="absolute -left-4 top-1/2 -translate-y-1/2 w-[1px] h-10 bg-zinc-800" />
                            <p className="text-[10px] font-black uppercase tracking-widest text-zinc-600 font-bold text-emerald-500 underline decoration-dotted underline-offset-4">Local NSE</p>
                            <p className="text-xl font-black text-emerald-500">₹{item.local_price || "0.0"}</p>
                        </div>
                    </div>

                    <div className="mt-8 pt-8 border-t border-zinc-800/40 flex items-center justify-between">
                        <div className="flex items-center gap-3">
                            <div className={`p-2 rounded-lg ${parseFloat(item.arb_yield) > 0.5 ? "bg-emerald-500/10 text-emerald-500" : "bg-zinc-800 text-zinc-600"}`}>
                                <TrendingUp className="w-4 h-4" />
                            </div>
                            <p className="text-xs font-bold text-zinc-400">
                                Action: <span className="text-zinc-200 uppercase">{item.action_label || "Hold"}</span>
                            </p>
                        </div>
                        <button className="text-xs font-black uppercase tracking-widest text-blue-500 hover:tracking-[0.2em] transition-all flex items-center gap-2">
                            Trade Edge
                            <ChevronRight className="w-4 h-4" />
                        </button>
                    </div>
                </div>
            ))}
        </div>
      </div>
    </div>
  );
};
