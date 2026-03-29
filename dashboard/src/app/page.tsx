"use client";

import React, { useState, useEffect } from "react";
import { 
  LayoutDashboard, 
  Search, 
  ShieldCheck, 
  Globe, 
  Bot, 
  History, 
  Settings,
  Bell,
  Menu,
  X,
  CreditCard,
  TrendingUp,
  Activity,
  AlertTriangle
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

const Sidebar = ({ activeTab, setActiveTab, config, setConfig }: { 
  activeTab: string, 
  setActiveTab: (tab: string) => void,
  config: any,
  setConfig: (c: any) => void
}) => {
  const menuItems = [
    { id: "scanner", label: "Live Scanner", icon: Search },
    { id: "mflab", label: "MF Consistency Lab", icon: ShieldCheck },
    { id: "commodities", label: "Commodities Terminal", icon: Globe },
    { id: "algos", label: "Options Algos", icon: Bot },
    { id: "history", label: "Scan History", icon: History },
  ];

  return (
    <div className="w-64 bg-[#0a0a0a] border-r border-zinc-800 h-screen flex flex-col fixed left-0 top-0 z-50">
      <div className="p-6">
        <div className="flex items-center gap-3 mb-8">
          <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
            <ShieldCheck className="text-white w-5 h-5" />
          </div>
          <h1 className="text-xl font-black bg-gradient-to-r from-blue-400 via-white to-zinc-500 bg-clip-text text-transparent tracking-tighter uppercase">
            Fortress Terminal
          </h1>
        </div>
        
        <nav className="space-y-1">
          {menuItems.map((item) => (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all ${
                activeTab === item.id 
                  ? "bg-blue-600/10 text-blue-400 border border-blue-500/20" 
                  : "text-zinc-500 hover:text-zinc-300 hover:bg-zinc-900"
              }`}
            >
              <item.icon className="w-5 h-5" />
              <span className="font-medium">{item.label}</span>
            </button>
          ))}
        </nav>

        {activeTab !== "dashboard" && (
          <motion.div 
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            className="mt-8 space-y-6 border-t border-zinc-800 pt-8"
          >
            <div>
              <label className="text-[10px] font-black uppercase tracking-widest text-zinc-600 mb-3 block">Portfolio & Risk</label>
              <div className="space-y-4">
                <div className="space-y-1">
                  <div className="flex justify-between text-[10px] font-bold text-zinc-400">
                    <span>Portfolio (₹)</span>
                    <span className="text-blue-400">{(config.portfolio_val / 10000000).toFixed(1)} Cr</span>
                  </div>
                  <input 
                    type="range" min="100000" max="100000000" step="500000" 
                    value={config.portfolio_val}
                    onChange={(e) => setConfig({...config, portfolio_val: Number(e.target.value)})}
                    className="w-full accent-blue-600 bg-zinc-800 rounded-lg h-1 appearance-none cursor-pointer" 
                  />
                </div>
                <div className="space-y-1">
                  <div className="flex justify-between text-[10px] font-bold text-zinc-400">
                    <span>Risk (%)</span>
                    <span className="text-blue-400">{config.risk_pct}%</span>
                  </div>
                  <input 
                    type="range" min="0.1" max="5.0" step="0.1" 
                    value={config.risk_pct}
                    onChange={(e) => setConfig({...config, risk_pct: Number(e.target.value)})}
                    className="w-full accent-blue-600 bg-zinc-800 rounded-lg h-1 appearance-none cursor-pointer" 
                  />
                </div>
              </div>
            </div>

            <div>
              <label className="text-[10px] font-black uppercase tracking-widest text-zinc-600 mb-3 block">Engine Weights</label>
              <div className="space-y-3">
                {Object.keys(config.weights).map((w) => (
                  <div key={w} className="space-y-1">
                    <div className="flex justify-between text-[10px] font-bold text-zinc-500">
                      <span className="capitalize">{w}</span>
                      <span className="text-zinc-300">{config.weights[w]}%</span>
                    </div>
                    <input 
                      type="range" min="0" max="100" 
                      value={config.weights[w]}
                      onChange={(e) => setConfig({
                        ...config, 
                        weights: {...config.weights, [w]: Number(e.target.value)}
                      })}
                      className="w-full accent-zinc-500 bg-zinc-800 rounded-lg h-1 appearance-none cursor-pointer" 
                    />
                  </div>
                ))}
              </div>
            </div>

            <div>
              <label className="text-[10px] font-black uppercase tracking-widest text-zinc-600 mb-3 block">Quality Gates</label>
              <div className="space-y-2">
                <div 
                    onClick={() => setConfig({...config, liquidity_gate: !config.liquidity_gate})}
                    className={`flex items-center justify-between p-2 rounded-lg border cursor-pointer transition-all ${config.liquidity_gate ? "bg-blue-600/10 border-blue-500/30" : "bg-zinc-900/50 border-zinc-800/50 opacity-50"}`}
                >
                  <span className="text-[10px] font-bold text-zinc-400">Liquidity (₹8Cr)</span>
                  <div className={`w-6 h-3 rounded-full flex items-center px-0.5 transition-colors ${config.liquidity_gate ? "bg-blue-600" : "bg-zinc-700"}`}><div className={`w-2 h-2 bg-white rounded-full transition-all ${config.liquidity_gate ? "ml-auto" : "ml-0"}`} /></div>
                </div>
                <div 
                    onClick={() => setConfig({...config, mcap_gate: !config.mcap_gate})}
                    className={`flex items-center justify-between p-2 rounded-lg border cursor-pointer transition-all ${config.mcap_gate ? "bg-blue-600/10 border-blue-500/30" : "bg-zinc-900/50 border-zinc-800/50 opacity-50"}`}
                >
                  <span className="text-[10px] font-bold text-zinc-400">MCap ({">"}₹1500Cr)</span>
                  <div className={`w-6 h-3 rounded-full flex items-center px-0.5 transition-colors ${config.mcap_gate ? "bg-blue-600" : "bg-zinc-700"}`}><div className={`w-2 h-2 bg-white rounded-full transition-all ${config.mcap_gate ? "ml-auto" : "ml-0"}`} /></div>
                </div>
              </div>
            </div>

            <div>
              <label className="text-[10px] font-black uppercase tracking-widest text-zinc-600 mb-3 block">Broker Selection</label>
              <div className="space-y-2">
                <select
                  value={config.broker}
                  onChange={(e) => setConfig({...config, broker: e.target.value})}
                  className="w-full bg-zinc-950 border border-zinc-800 rounded-xl py-3 px-4 text-xs font-bold text-white focus:ring-2 focus:ring-blue-500/20 focus:outline-none transition-all"
                >
                  <option value="Zerodha">Zerodha</option>
                  <option value="Dhan">Dhan</option>
                </select>
              </div>
            </div>
          </motion.div>
        )}
      </div>
      
      <div className="mt-auto p-6 border-t border-zinc-800">
        <button className="w-full flex items-center gap-3 px-4 py-3 text-zinc-500 hover:text-zinc-300 transition-all">
          <Settings className="w-5 h-5" />
          <span className="font-medium">Settings</span>
        </button>
      </div>
    </div>
  );
};

const Header = () => {
  const [status, setStatus] = useState<string>("connecting");

  useEffect(() => {
    fetch("/api/health")
      .then(res => res.json())
      .then(data => setStatus(data.status))
      .catch(() => setStatus("offline"));
  }, []);

  return (
    <header className="h-16 border-b border-zinc-800 bg-black/50 backdrop-blur-xl flex items-center justify-between px-8 sticky top-0 z-40 ml-64">
      <div className="flex items-center gap-4">
        <span className="text-zinc-400 font-medium">Dashboard</span>
        <span className="text-zinc-600">/</span>
        <span className="text-zinc-200 font-semibold">Institutional Terminal</span>
      </div>
      
      <div className="flex items-center gap-6">
        <div className={`flex items-center gap-2 bg-zinc-900 border border-zinc-800 px-3 py-1.5 rounded-full`}>
          <div className={`w-2 h-2 rounded-full ${status === "healthy" ? "bg-emerald-500 animate-pulse" : "bg-rose-500"}`} />
          <span className={`${status === "healthy" ? "text-emerald-500" : "text-rose-500"} text-xs font-bold uppercase tracking-wider`}>
            {status === "healthy" ? "Engine Live" : status === "connecting" ? "Connecting..." : "Engine Offline"}
          </span>
        </div>
        
        <div className="h-8 w-[1px] bg-zinc-800" />
        
        <button className="relative text-zinc-400 hover:text-white transition-colors">
          <Bell className="w-5 h-5" />
          <span className="absolute -top-1 -right-1 w-2 h-2 bg-blue-600 rounded-full" />
        </button>
        
        <div className="flex items-center gap-3 bg-blue-600/10 hover:bg-blue-600/20 cursor-pointer px-4 py-2 rounded-xl border border-blue-500/20 transition-all group">
          <div className="flex flex-col items-end">
            <span className="text-[10px] font-black uppercase tracking-widest text-blue-400">Security Clearance</span>
            <span className="text-xs font-bold text-white group-hover:text-blue-400 transition-colors">LVL 4: ADMIN</span>
          </div>
          <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center text-white text-xs font-bold shadow-lg shadow-blue-500/20">
            <ShieldCheck className="w-4 h-4" />
          </div>
        </div>
      </div>
    </header>
  );
};

const StatCard = ({ icon: Icon, label, value, trend, color }: any) => (
  <div className="bg-zinc-900/50 border border-zinc-800 p-6 rounded-3xl hover:border-zinc-700 transition-all">
    <div className="flex items-center justify-between mb-4">
      <div className={`p-3 rounded-2xl bg-${color}-500/10 text-${color}-400`}>
        <Icon className="w-6 h-6" />
      </div>
      {trend && (
        <div className={`flex items-center gap-1 text-sm font-bold ${trend > 0 ? "text-emerald-400" : "text-rose-400"}`}>
          {trend > 0 ? "↑" : "↓"}{Math.abs(trend)}%
        </div>
      )}
    </div>
    <div className="space-y-1">
      <p className="text-zinc-500 text-sm font-medium">{label}</p>
      <h3 className="text-2xl font-bold text-white">{value}</h3>
    </div>
  </div>
);

import { LiveScanner } from "@/components/LiveScanner";
import { MFTerminal } from "@/components/MFTerminal";
import { CommoditiesTerminal } from "@/components/CommoditiesTerminal";

const LandingDashboard = () => (
  <div className="space-y-8 p-8 ml-64 min-h-screen bg-black text-white">
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
    >
      <StatCard icon={TrendingUp} label="Total Assets Under Analysis" value="₹12.4 Cr" trend={12.5} color="blue" />
      <StatCard icon={Activity} label="Active Scans Today" value="1,482" trend={5.2} color="emerald" />
      <StatCard icon={AlertTriangle} label="High Conviction Setups" value="24" trend={-2.1} color="amber" />
      <StatCard icon={CreditCard} label="Available Capital Weight" value="84%" trend={3.4} color="purple" />
    </motion.div>

    <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
      <div className="lg:col-span-2 bg-zinc-900/40 border border-zinc-800/60 rounded-[32px] p-10 overflow-hidden relative group">
        <div className="flex items-center justify-between mb-8">
          <h3 className="text-2xl font-black flex items-center gap-3">
            <LayoutDashboard className="text-blue-500" />
            Market Pulse
          </h3>
          <div className="flex items-center gap-4">
              <span className="text-xs font-bold text-zinc-500 uppercase tracking-widest">Nifty 50 Index</span>
              <div className="w-12 h-6 bg-emerald-500/20 rounded-full flex items-center px-1">
                  <motion.div 
                    animate={{ x: [0, 24, 0] }}
                    transition={{ repeat: Infinity, duration: 2 }}
                    className="w-4 h-4 bg-emerald-500 rounded-full shadow-[0_0_10px_rgba(16,185,129,0.5)]" 
                  />
              </div>
          </div>
        </div>
        
        <div className="h-72 flex flex-col items-center justify-center border border-dashed border-zinc-800 rounded-[32px] text-zinc-600 font-bold bg-zinc-950/20 group-hover:border-zinc-700 transition-all relative overflow-hidden">
            <div className="absolute inset-0 opacity-10">
                <div className="absolute inset-0 bg-gradient-to-t from-blue-500/20 to-transparent" />
                <div className="w-full h-full border-b border-zinc-800" style={{ backgroundSize: '40px 40px', backgroundImage: 'linear-gradient(to right, #1f1f1f 1px, transparent 1px), linear-gradient(to bottom, #1f1f1f 1px, transparent 1px)' }} />
            </div>
            <TrendingUp className="w-12 h-12 mb-4 opacity-10 group-hover:opacity-20 transition-opacity" />
            <p className="text-sm uppercase tracking-[0.2em] font-black group-hover:text-zinc-500 transition-colors">Institutional Index Stream Active</p>
        </div>
      </div>
      
      <div className="bg-zinc-900/20 border border-zinc-800/40 rounded-[32px] p-8 backdrop-blur-sm">
        <h3 className="text-lg font-black mb-8 flex items-center gap-3 text-white">
          <ShieldCheck className="text-emerald-500" />
          High Scored Picks
        </h3>
        
        <div className="space-y-3">
          {[
            { ticker: "RELIANCE", name: "RELIANCE.NS", score: 88, type: "Momentum" },
            { ticker: "TCS", name: "TCS.NS", score: 85, type: "Quality" },
            { ticker: "HDFCBANK", name: "HDFCBANK.NS", score: 82, type: "Recovery" },
            { ticker: "ZOMATO", name: "ZOMATO.NS", score: 81, type: "Alpha" }
          ].map((item, i) => (
            <div key={i} className="flex items-center justify-between p-4 bg-zinc-950/40 rounded-2xl border border-zinc-800/40 hover:border-emerald-500/30 transition-all cursor-pointer group">
              <div className="flex items-center gap-4">
                <div className="w-10 h-10 bg-zinc-900 rounded-xl flex items-center justify-center font-black text-[10px] group-hover:bg-blue-600 group-hover:text-white transition-all">
                  {item.ticker}
                </div>
                <div>
                  <p className="font-bold text-xs text-white">{item.name}</p>
                  <p className="text-zinc-600 text-[10px] font-black uppercase tracking-tighter">{item.type} Signal</p>
                </div>
              </div>
              <div className="text-right">
                <p className="text-emerald-400 font-black text-xs">{item.score}</p>
                <div className="w-8 h-1 bg-zinc-800 rounded-full mt-1 overflow-hidden">
                    <div className="h-full bg-emerald-500" style={{ width: `${item.score}%` }} />
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  </div>
);

export default function App() {
  const [activeTab, setActiveTab] = useState("dashboard");
  const [scannerConfig, setScannerConfig] = useState({
    portfolio_val: 1000000,
    risk_pct: 1.0,
    weights: { technical: 50, fundamental: 25, sentiment: 15, context: 10 },
    liquidity_gate: true,
    mcap_gate: true,
    broker: "Zerodha"
  });

  const renderContent = () => {
    switch (activeTab) {
      case "dashboard":
        return <LandingDashboard />;
      case "scanner":
        return (
          <motion.div 
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="p-8 ml-64 min-h-screen bg-black"
          >
            <div className="mb-8">
                <h2 className="text-4xl font-black text-white mb-2">Institutional Scanner</h2>
                <p className="text-zinc-500 font-bold uppercase tracking-widest text-xs">Fortress Core x FastAPI 2.0 Engine</p>
            </div>
            <LiveScanner config={scannerConfig} />
          </motion.div>
        );
      case "mflab":
        return (
          <motion.div 
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="p-8 ml-64 min-h-screen bg-black"
          >
            <div className="mb-8">
                <h2 className="text-4xl font-black text-white mb-2">Mutual Fund Lab</h2>
                <p className="text-zinc-500 font-bold uppercase tracking-widest text-xs">Benchmarking Engine x Rolling Consistency</p>
            </div>
            <MFTerminal config={scannerConfig} />
          </motion.div>
        );
      case "commodities":
        return (
          <motion.div 
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="p-8 ml-64 min-h-screen bg-black"
          >
            <div className="mb-8">
                <h2 className="text-4xl font-black text-white mb-2">Commodities Terminal</h2>
                <p className="text-zinc-500 font-bold uppercase tracking-widest text-xs">Global-Local Price Parity x Arbitrage Edge</p>
            </div>
            <CommoditiesTerminal config={scannerConfig} />
          </motion.div>
        );
      default:
        return (
           <motion.div 
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="p-8 ml-64 min-h-screen bg-black flex flex-col items-center justify-center text-zinc-500"
          >
            <div className="w-16 h-16 border-4 border-zinc-800 border-t-blue-500 rounded-full animate-spin mb-4" />
            <p className="font-bold tracking-widest uppercase">Initializing Protocol {activeTab.toUpperCase()}...</p>
          </motion.div>
        );
    }
  };

  return (
    <main className="min-h-screen bg-black">
      <Sidebar 
        activeTab={activeTab} 
        setActiveTab={setActiveTab} 
        config={scannerConfig}
        setConfig={setScannerConfig}
      />
      <Header />
      
      <AnimatePresence mode="wait">
        {renderContent()}
      </AnimatePresence>
    </main>
  );
}
