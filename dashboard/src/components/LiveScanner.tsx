"use client";

import React, { useState, useEffect, useMemo } from "react";
import { 
  Search, 
  TrendingUp, 
  ShieldCheck, 
  AlertCircle, 
  Loader2,
  ChevronRight,
  BarChart4,
  LayoutDashboard,
  Zap,
  Target,
  Download,
  Filter,
  Eye
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

const ALL_COLUMNS: any = {
    "Symbol": { label: "Symbol" },
    "Verdict": { label: "Verdict" },
    "Actions": { label: "Trade Now" },
    "Score": { label: "Conviction" },
    "Price": { label: "Price ₹" },
    "RSI": { label: "RSI" },
    "News": { label: "News" },
    "Events": { label: "Events" },
    "Sector": { label: "Sector" },
    "Position_Qty": { label: "Qty" },
    "Stop_Loss": { label: "SL Price" },
    "Target_10D": { label: "10D Target" },
    "Analysts": { label: "Analyst Count" },
    "Tgt_High": { label: "High Target" },
    "Tgt_Median": { label: "Median Target" },
    "Tgt_Low": { label: "Low Target" },
    "Tgt_Mean": { label: "Mean Target" },
    "Dispersion_Alert": { label: "Dispersion" },
    "Ret_30D": { label: "30D Backtest" },
    "Ret_60D": { label: "60D Backtest" },
    "Ret_90D": { label: "90D Backtest" },
    "Strategy": { label: "Strategy" },
    "Buy_Zone": { label: "Buy Zone" },
    "Steam_Left": { label: "Steam Left ₹" },
    "Days_To_Target": { label: "Est. Days" },
    "Resilience": { label: "War/News Resilience" },
    "Gap_Integrity": { label: "Gap Integrity" },
    "Velocity": { label: "Momentum Velocity" },
    "Ret_7D": { label: "7D Return" },
    "Technical_Score": { label: "Technical" },
    "Tech_Score": { label: "Tech Score" },
    "Fundamental_Score": { label: "Fundamental" },
    "Fund_Score": { label: "Fund Score" },
    "Sentiment_Score": { label: "Sentiment" },
    "Sent_Score": { label: "Sent Score" },
    "Context_Score": { label: "Context/RS/MTF" },
    "Score_Pre_Regime": { label: "Pre-Regime Score" },
    "Regime": { label: "Market Regime" },
    "Market_Regime": { label: "Market Regime" },
    "Regime_Tag": { label: "Regime Tag" },
    "Regime_Multiplier": { label: "Regime x" },
    "India_VIX": { label: "India VIX" },
    "RS_Rank": { label: "RS Rank %" },
    "RS_3M": { label: "RS 3M" },
    "RS_6M": { label: "RS 6M" },
    "RS_12M": { label: "RS 12M" },
    "RS_Composite": { label: "RS Composite" },
    "Vol_Adj_Mom": { label: "Vol Adj Mom" },
    "EMA200_Extension_Pct": { label: "EMA200 Extension" },
    "Quality_Gate_Failures": { label: "Quality Gates" },
    "Quality_Gate_Pass": { label: "Quality Pass" },
    "Avoid_Flag": { label: "Avoid" }
};

export const LiveScanner = ({ config }: { config: any }) => {
  const [universe, setUniverse] = useState("Nifty 50");
  const [scanning, setScanning] = useState(false);
  const [results, setResults] = useState<any[]>([]);
  const [sectorPulse, setSectorPulse] = useState<any[]>([]);
  const [universes, setUniverses] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [activeColumns, setActiveColumns] = useState<string[]>(["Symbol", "Score", "Price", "Technical_Score", "Fundamental_Score", "Sector", "Strategy", "Verdict", "RSI", "Velocity", "Actions"]);
  const [showColPicker, setShowColPicker] = useState(false);
  const [searchTerm, setSearchTerm] = useState("");

  useEffect(() => {
    fetch("/api/universes")
      .then(res => res.json())
      .then(data => setUniverses(data))
      .catch(err => console.error("Error fetching universes", err));
  }, []);

  const fetchSectorPulse = async (uni: string) => {
    try {
      const res = await fetch(`/api/sector-pulse?universe=${encodeURIComponent(uni)}`);
      const data = await res.json();
      setSectorPulse(data);
    } catch (e) {
      console.error("Pulse fetch failed", e);
    }
  };

  const runScan = async () => {
    setScanning(true);
    setError(null);
    try {
      const resp = await fetch("/api/scan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          universe,
          portfolio_val: config.portfolio_val,
          risk_pct: config.risk_pct / 100,
          weights: config.weights,
          enable_regime: true,
          liquidity_cr_min: config.liquidity_gate ? 8.0 : 0.0,
          market_cap_cr_min: config.mcap_gate ? 1500.0 : 0.0,
          price_min: 80.0,
          broker: config.broker
        })
      });
      if (!resp.ok) throw new Error("Scan failed");
      const data = await resp.json();
      setResults(data);
      fetchSectorPulse(universe);
    } catch (err: any) {
      setError(err.message || "An unexpected error occurred");
    } finally {
      setScanning(false);
    }
  };

  const exportCSV = () => {
    if (results.length === 0) return;
    const headers = Object.keys(results[0]).join(",");
    const rows = results.map(row => Object.values(row).join(",")).join("\n");
    const blob = new Blob([headers + "\n" + rows], { type: "text/csv" });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `Fortress_Scan_${universe.replace(" ", "_")}.csv`;
    a.click();
  };

  const filteredResults = useMemo(() => {
    return results.filter(r => 
        r.Symbol.toLowerCase().includes(searchTerm.toLowerCase()) ||
        r.Sector.toLowerCase().includes(searchTerm.toLowerCase())
    );
  }, [results, searchTerm]);

  const momentumPicks = useMemo(() => filteredResults.filter(r => r.Strategy === "Momentum Pick" && r.Score >= 60), [filteredResults]);
  const longTermPicks = useMemo(() => filteredResults.filter(r => r.Strategy === "Long-Term Pick" && r.Score >= 60), [filteredResults]);

  return (
    <div className="space-y-8">
      {/* Search & Control Header */}
      <div className="flex flex-col xl:flex-row gap-4 items-end justify-between bg-zinc-900/40 p-6 rounded-3xl border border-zinc-800/50 backdrop-blur-sm">
        <div className="flex flex-col md:flex-row gap-4 flex-grow w-full md:w-auto">
            <div className="space-y-2 flex-grow max-w-xs">
                <label className="text-[10px] font-black uppercase tracking-widest text-zinc-500 ml-1">Market Universe</label>
                <div className="relative">
                    <Filter className="absolute left-4 top-1/2 -translate-y-1/2 text-zinc-500 w-4 h-4" />
                    <select 
                        value={universe}
                        onChange={(e) => setUniverse(e.target.value)}
                        className="w-full bg-zinc-950 border border-zinc-800 rounded-xl py-3 pl-10 pr-4 text-xs font-bold text-white focus:ring-2 focus:ring-blue-500/20 focus:outline-none transition-all appearance-none cursor-pointer"
                    >
                    {universes.length > 0 ? universes.map(u => (
                        <option key={u} value={u}>{u}</option>
                    )) : (
                        <option>Loading...</option>
                    )}
                    </select>
                </div>
            </div>

            <div className="space-y-2 flex-grow max-w-sm">
                <label className="text-[10px] font-black uppercase tracking-widest text-zinc-500 ml-1">Symbol Search</label>
                <div className="relative">
                    <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-zinc-500 w-4 h-4" />
                    <input 
                        type="text"
                        placeholder="Search tickers or sectors..."
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        className="w-full bg-zinc-950 border border-zinc-800 rounded-xl py-3 pl-10 pr-4 text-xs font-bold text-white focus:ring-2 focus:ring-blue-500/20 focus:outline-none transition-all"
                    />
                </div>
            </div>
        </div>
        
        <div className="flex items-center gap-3 w-full xl:w-auto">
            <button 
                onClick={() => setShowColPicker(!showColPicker)}
                className="flex items-center gap-2 px-6 py-3 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded-xl font-bold text-xs transition-all border border-zinc-700/50"
            >
                <Eye className="w-4 h-4" />
                Columns
            </button>

            <button 
                onClick={exportCSV}
                className="flex items-center gap-2 px-6 py-3 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded-xl font-bold text-xs transition-all border border-zinc-700/50"
            >
                <Download className="w-4 h-4" />
                Export
            </button>

            <button 
                onClick={runScan}
                disabled={scanning}
                className={`flex-grow xl:flex-grow-0 flex items-center justify-center gap-3 px-10 py-3 rounded-xl font-black text-xs uppercase tracking-widest transition-all ${
                    scanning 
                    ? "bg-blue-600/20 text-blue-400 cursor-not-allowed border border-blue-500/20" 
                    : "bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-500/20 active:scale-95"
                }`}
            >
            {scanning ? (
                <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Analyzing...
                </>
            ) : (
                <>
                <ShieldCheck className="w-4 h-4" />
                Execute Scan
                </>
            )}
            </button>
        </div>
      </div>

      <AnimatePresence>
        {showColPicker && (
            <motion.div 
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="bg-zinc-900 border border-zinc-800 p-6 rounded-3xl grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3"
            >
                {Object.keys(ALL_COLUMNS).map(col => (
                    <button 
                        key={col}
                        onClick={() => {
                            if (activeColumns.includes(col)) setActiveColumns(activeColumns.filter(c => c !== col));
                            else setActiveColumns([...activeColumns, col]);
                        }}
                        className={`px-3 py-2 rounded-lg text-[10px] font-black uppercase tracking-tighter transition-all border ${
                            activeColumns.includes(col) 
                            ? "bg-blue-600/10 border-blue-500/50 text-blue-400" 
                            : "bg-zinc-950 border-zinc-800 text-zinc-600 hover:text-zinc-400"
                        }`}
                    >
                        {ALL_COLUMNS[col].label}
                    </button>
                ))}
            </motion.div>
        )}
      </AnimatePresence>

      {/* Results Workspace */}
      <div className="space-y-12">
        {/* Sector Intelligence Table */}
        {sectorPulse.length > 0 && !scanning && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-4">
                <div className="flex items-center gap-3 ml-2">
                    <Zap className="text-amber-500 w-5 h-5" />
                    <h3 className="text-xl font-black text-white uppercase tracking-tighter">Sector Intelligence</h3>
                </div>
                <div className="bg-zinc-900/30 border border-zinc-800/50 rounded-[32px] overflow-hidden backdrop-blur-md">
                    <table className="w-full text-left border-collapse">
                        <thead>
                            <tr className="bg-zinc-950/40 border-b border-zinc-800/60">
                        <th className="px-8 py-4 text-[10px] font-black uppercase tracking-widest text-zinc-400">Sector</th>
                                <th className="px-8 py-4 text-[10px] font-black uppercase tracking-widest text-zinc-400">Thesis</th>
                                <th className="px-8 py-4 text-[10px] font-black uppercase tracking-widest text-zinc-400 text-center">Velocity</th>
                                <th className="px-8 py-4 text-[10px] font-black uppercase tracking-widest text-zinc-400 text-center">Inst. Breadth</th>
                                <th className="px-8 py-4 text-[10px] font-black uppercase tracking-widest text-zinc-400 text-center">On the Rise</th>
                                <th className="px-8 py-4 text-[10px] font-black uppercase tracking-widest text-zinc-400 text-center">On the Fall</th>
                                <th className="px-8 py-4 text-[10px] font-black uppercase tracking-widest text-zinc-400 text-right">Avg Score</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-zinc-800/40">
                            {sectorPulse.sort((a,b) => b.Velocity - a.Velocity).map((s, i) => (
                                <tr key={i} className="hover:bg-zinc-800/80 transition-all group border-b border-zinc-800/20">
                                    <td className="px-8 py-4 text-sm font-black text-white uppercase tracking-tight">{s.Sector}</td>
                                    <td className="px-8 py-4">
                                        <div className={`text-xs font-black uppercase tracking-widest ${s.Velocity > 1 ? "text-emerald-400" : s.Velocity < -1 ? "text-rose-400" : "text-zinc-300"}`}>
                                            {s.Thesis}
                                        </div>
                                    </td>
                                    <td className="px-8 py-4 text-center font-mono font-bold text-xs text-zinc-400">{s.Velocity}%</td>
                                    <td className="px-8 py-4 text-center">
                                        <div className="w-24 h-1.5 bg-zinc-800 rounded-full mx-auto overflow-hidden">
                                            <div className="h-full bg-blue-500" style={{ width: `${s.Breadth}%` }} />
                                        </div>
                                        <p className="text-[8px] font-black text-zinc-600 uppercase mt-1">{s.Breadth}% Breadth</p>
                                    </td>
                                    <td className="px-8 py-4 text-center">
                                        <span className={`text-xs font-black ${s.On_the_Rise ? "text-emerald-500" : "text-zinc-600"}`}>
                                            {s.On_the_Rise || "-"}
                                        </span>
                                    </td>
                                    <td className="px-8 py-4 text-center">
                                        <span className={`text-xs font-black ${s.On_the_Fall ? "text-rose-500" : "text-zinc-600"}`}>
                                            {s.On_the_Fall || "-"}
                                        </span>
                                    </td>
                                    <td className="px-8 py-4 text-right">
                                        <span className={`text-sm font-black ${s.Avg_Score >= 70 ? "text-emerald-500" : "text-zinc-500"}`}>
                                            {s.Avg_Score.toFixed(1)}
                                        </span>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </motion.div>
        )}

        {/* Strategic Picks */}
        {!scanning && results.length > 0 && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div className="space-y-4">
                    <div className="flex items-center gap-3 ml-2">
                        <TrendingUp className="text-blue-500 w-5 h-5" />
                        <h3 className="text-lg font-black text-white uppercase tracking-tighter">🚀 Momentum Picks ({momentumPicks.length})</h3>
                    </div>
                    <div className="bg-zinc-900/30 border border-zinc-800/50 rounded-3xl overflow-hidden p-2">
                        {momentumPicks.length > 0 ? momentumPicks.slice(0, 5).map((p, i) => (
                            <div key={i} className="flex items-center justify-between p-4 hover:bg-zinc-900/40 rounded-2xl transition-all">
                                <div className="flex items-center gap-3">
                                    <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center font-black text-[12px] shadow-sm shadow-blue-900/50">{p.Symbol.substring(0,2).toUpperCase()}</div>
                                    <span className="font-bold text-xs text-white">{p.Symbol}</span>
                                </div>
                                <span className="font-black text-emerald-400 text-sm">{p.Score}</span>
                            </div>
                        )) : (
                            <div className="p-8 text-center text-zinc-600 text-xs font-bold uppercase italic">No active momentum signals</div>
                        )}
                    </div>
                </div>

                <div className="space-y-4">
                    <div className="flex items-center gap-3 ml-2">
                        <Target className="text-purple-500 w-5 h-5" />
                        <h3 className="text-lg font-black text-white uppercase tracking-tighter">💎 Long-Term Picks ({longTermPicks.length})</h3>
                    </div>
                    <div className="bg-zinc-900/30 border border-zinc-800/50 rounded-3xl overflow-hidden p-2">
                        {longTermPicks.length > 0 ? longTermPicks.slice(0, 5).map((p, i) => (
                            <div key={i} className="flex items-center justify-between p-4 hover:bg-zinc-900/40 rounded-2xl transition-all">
                                <div className="flex items-center gap-3">
                                    <div className="w-8 h-8 bg-purple-600 rounded-lg flex items-center justify-center font-black text-[12px] shadow-sm shadow-purple-900/50">{p.Symbol.substring(0,2).toUpperCase()}</div>
                                    <span className="font-bold text-xs text-white">{p.Symbol}</span>
                                </div>
                                <span className="font-black text-purple-400 text-sm">{p.Score}</span>
                            </div>
                        )) : (
                            <div className="p-8 text-center text-zinc-600 text-xs font-bold uppercase italic">No active structural signals</div>
                        )}
                    </div>
                </div>
            </div>
        )}

        {/* Dynamic Data Terminal */}
        <div className="min-h-[400px]">
            <AnimatePresence mode="wait">
            {scanning ? (
                <motion.div 
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="flex flex-col items-center justify-center p-20 space-y-4"
                >
                    <div className="w-16 h-16 border-4 border-blue-600/20 border-t-blue-600 rounded-full animate-spin" />
                    <div className="text-center">
                        <h3 className="text-lg font-bold text-white">Full Engine Scan In Progress</h3>
                        <p className="text-zinc-500 text-sm">Processing {universe} with advanced weights...</p>
                    </div>
                </motion.div>
            ) : error ? (
                <motion.div className="bg-rose-500/10 border border-rose-500/20 p-8 rounded-3xl text-rose-400">
                    <AlertCircle className="mb-2" />
                    <h3 className="font-bold">Scan Engine Failed</h3>
                    <p className="text-xs">{error}</p>
                </motion.div>
            ) : results.length > 0 ? (
            <motion.div 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-zinc-900/30 border border-zinc-800/50 rounded-[32px] overflow-hidden backdrop-blur-md"
                >
                <div className="overflow-x-auto">
                    <table className="w-full text-left border-collapse whitespace-nowrap">
                        <thead>
                        <tr className="bg-zinc-950/80 border-b border-zinc-800/60">
                            {activeColumns.map(col => (
                                <th key={col} className={`px-6 py-5 text-[10px] font-black uppercase tracking-widest text-zinc-400 bg-zinc-950/80 backdrop-blur-xl ${col === "Score" || col === "Price" ? "text-right" : ""} ${col === "Symbol" ? "sticky left-0 z-20 border-r border-zinc-800/50 shadow-[4px_0_24px_-8px_rgba(0,0,0,0.5)]" : ""}`}>
                                    {ALL_COLUMNS[col]?.label || col}
                                </th>
                            ))}
                        </tr>
                        </thead>
                        <tbody className="divide-y divide-zinc-800/60">
                        {filteredResults.map((item, idx) => (
                            <tr key={idx} className="hover:bg-zinc-800/80 transition-colors group cursor-pointer bg-black/20">
                            {activeColumns.map(col => (
                                <td key={col} className={`px-6 py-5 ${col === "Score" || col === "Price" ? "text-right" : ""} ${col === "Symbol" ? "sticky left-0 z-10 bg-zinc-950 border-r border-zinc-800/50 shadow-[4px_0_24px_-8px_rgba(0,0,0,0.5)] group-hover:bg-zinc-800" : ""}`}>
                                    {col === "Symbol" ? (
                                        <div className="flex items-center gap-4">
                                            <div className="w-10 h-10 bg-zinc-800 rounded-xl flex items-center justify-center font-black text-xs border border-zinc-700 shadow-lg group-hover:bg-blue-600 group-hover:border-blue-500 transition-colors">
                                                {item.Symbol.substring(0,2).toUpperCase()}
                                            </div>
                                            <span className="font-bold text-white text-sm">{item.Symbol}</span>
                                        </div>
                                    ) : col === "Score" ? (
                                        <span className={`text-lg font-black ${item.Score >= 80 ? "text-emerald-500 shadow-emerald-500/20" : item.Score >= 60 ? "text-blue-400" : "text-zinc-300"}`}>{item.Score}</span>
                                    ) : col === "Price" ? (
                                        <span className="font-mono text-sm font-bold text-zinc-200">₹{item.Price?.toLocaleString()}</span>
                                    ) : col === "Technical_Score" || col === "Fundamental_Score" ? (
                                        <div className="flex items-center gap-3">
                                            <div className="w-16 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                                                <div className={`h-full ${col.startsWith("Tech") ? "bg-blue-500" : "bg-emerald-500"}`} style={{ width: `${item[col]}%` }} />
                                            </div>
                                            <span className="text-xs font-bold text-zinc-300">{item[col]}</span>
                                        </div>
                                    ) : col === "Actions" ? (
                                        <div dangerouslySetInnerHTML={{ __html: item[col] }} />
                                    ) : (
                                        <span className="text-sm font-medium text-zinc-300">{String(item[col] || "-")}</span>
                                    )}
                                </td>
                            ))}
                            </tr>
                        ))}
                        </tbody>
                    </table>
                </div>
                </motion.div>
            ) : (
                <div className="flex flex-col items-center justify-center p-20 text-zinc-600 bg-zinc-900/20 rounded-[40px] border border-zinc-800/20 border-dashed">
                    <BarChart4 className="w-12 h-12 mb-4 opacity-10" />
                    <p className="font-bold tracking-tight uppercase text-xs">Awaiting System Initialization</p>
                </div>
            )}
            </AnimatePresence>
        </div>
      </div>
    </div>
  );
};
