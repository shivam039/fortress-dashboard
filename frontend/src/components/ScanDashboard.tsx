import React, { useState, useEffect } from 'react';
import { fetchUniverses, startScan, socket, ScanResult } from '../services/api';
import { Download } from 'lucide-react';
import clsx from 'clsx';

const COLUMNS = [
  { key: 'Symbol', label: 'Symbol' },
  { key: 'Verdict', label: 'Verdict' },
  { key: 'Score', label: 'Conviction' },
  { key: 'Price', label: 'Price ₹' },
  { key: 'RSI', label: 'RSI' },
  { key: 'News', label: 'News' },
  { key: 'Events', label: 'Events' },
  { key: 'Sector', label: 'Sector' },
  { key: 'Position_Qty', label: 'Qty' },
  { key: 'Stop_Loss', label: 'SL Price' },
  { key: 'Target_10D', label: '10D Target' },
  { key: 'Analysts', label: 'Analysts' },
  { key: 'Dispersion_Alert', label: 'Dispersion' },
  { key: 'Ret_30D', label: '30D Backtest' },
];

const ScanDashboard: React.FC = () => {
  const [universes, setUniverses] = useState<string[]>([]);
  const [selectedUniverse, setSelectedUniverse] = useState<string>('');
  const [portfolioVal, setPortfolioVal] = useState(1000000);
  const [riskPct, setRiskPct] = useState(1);
  const [scanning, setScanning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusMsg, setStatusMsg] = useState('');
  const [results, setResults] = useState<ScanResult[]>([]);
  const [selectedColumns, setSelectedColumns] = useState<string[]>(COLUMNS.map(c => c.key));

  useEffect(() => {
    fetchUniverses().then(data => {
      setUniverses(data);
      if (data.length > 0) setSelectedUniverse(data[0]);
    });

    socket.on('scan-progress', (data: any) => {
      if (data.universe === selectedUniverse || !selectedUniverse) {
        setProgress(data.progress);
        setStatusMsg(data.message);
        if (data.result) {
          setResults(prev => {
            // Avoid duplicates if re-rendering or socket spam
            if (prev.find(r => r.Symbol === data.result.Symbol)) return prev;
            return [...prev, data.result].sort((a, b) => b.Score - a.Score);
          });
        }
      }
    });

    socket.on('scan-complete', (data: any) => {
      setScanning(false);
      setStatusMsg(data.message);
      setProgress(100);
    });

    socket.on('scan-error', (data: any) => {
        setScanning(false);
        setStatusMsg(`Error: ${data.message}`);
    });

    return () => {
      socket.off('scan-progress');
      socket.off('scan-complete');
      socket.off('scan-error');
    };
  }, [selectedUniverse]);

  const handleScan = async () => {
    if (!selectedUniverse) return;
    setScanning(true);
    setResults([]);
    setProgress(0);
    setStatusMsg('Initializing...');
    await startScan(selectedUniverse, portfolioVal, riskPct / 100);
  };

  const exportCSV = () => {
      if (results.length === 0) return;
      const headers = selectedColumns.join(',');
      const rows = results.map(r => selectedColumns.map(col => (r as any)[col]).join(',')).join('\n');
      const csv = `${headers}\n${rows}`;
      const blob = new Blob([csv], { type: 'text/csv' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `Fortress_Trades_${new Date().toISOString().slice(0,10)}.csv`;
      a.click();
  };

  return (
    <div className="p-6 space-y-6">
      <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
        <h2 className="text-xl font-bold mb-4 text-slate-800">🚀 Live Scanner</h2>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-6">
          <div>
            <label className="block text-sm font-medium text-slate-600 mb-1">Select Index</label>
            <select
              className="w-full border rounded-lg p-2 bg-slate-50"
              value={selectedUniverse}
              onChange={e => setSelectedUniverse(e.target.value)}
              disabled={scanning}
            >
              {universes.map(u => <option key={u} value={u}>{u}</option>)}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-600 mb-1">Portfolio Value (₹)</label>
            <input
              type="number"
              className="w-full border rounded-lg p-2 bg-slate-50"
              value={portfolioVal}
              onChange={e => setPortfolioVal(Number(e.target.value))}
              disabled={scanning}
            />
          </div>
          <div>
             <label className="block text-sm font-medium text-slate-600 mb-1">Risk Per Trade (%)</label>
             <input
                type="range"
                min="0.5" max="3" step="0.1"
                className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer"
                value={riskPct}
                onChange={e => setRiskPct(Number(e.target.value))}
                disabled={scanning}
             />
             <span className="text-xs text-slate-500">{riskPct}%</span>
          </div>
          <div className="flex items-end">
             <button
               className={clsx(
                   "w-full py-2 px-4 rounded-lg font-bold text-white transition-colors",
                   scanning ? "bg-slate-400 cursor-not-allowed" : "bg-indigo-600 hover:bg-indigo-700"
               )}
               onClick={handleScan}
               disabled={scanning}
             >
               {scanning ? 'Scanning...' : 'EXECUTE SYSTEM SCAN'}
             </button>
          </div>
        </div>

        {scanning || progress > 0 ? (
            <div className="mb-6">
                <div className="flex justify-between text-sm mb-1">
                    <span>{statusMsg}</span>
                    <span>{Math.round(progress)}%</span>
                </div>
                <div className="w-full bg-slate-200 rounded-full h-2.5">
                    <div className="bg-indigo-600 h-2.5 rounded-full transition-all duration-300" style={{ width: `${progress}%` }}></div>
                </div>
            </div>
        ) : null}
      </div>

      {results.length > 0 && (
          <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
             <div className="flex justify-between items-center mb-4">
                 <h3 className="text-lg font-bold">Results ({results.length})</h3>
                 <button onClick={exportCSV} className="flex items-center gap-2 text-indigo-600 hover:text-indigo-800">
                     <Download size={18} /> Export CSV
                 </button>
             </div>

             <div className="overflow-x-auto">
                 <table className="min-w-full divide-y divide-slate-200">
                     <thead className="bg-slate-50">
                         <tr>
                             {COLUMNS.filter(c => selectedColumns.includes(c.key)).map(col => (
                                 <th key={col.key} className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">
                                     {col.label}
                                 </th>
                             ))}
                         </tr>
                     </thead>
                     <tbody className="bg-white divide-y divide-slate-200">
                         {results.map((row, idx) => (
                             <tr key={idx} className={idx % 2 === 0 ? 'bg-white' : 'bg-slate-50'}>
                                 {COLUMNS.filter(c => selectedColumns.includes(c.key)).map(col => (
                                     <td key={col.key} className="px-6 py-4 whitespace-nowrap text-sm text-slate-700">
                                         {col.key === 'Score' ? (
                                             <div className="w-24 bg-slate-200 rounded-full h-2.5">
                                                 <div
                                                    className={clsx("h-2.5 rounded-full", (row as any)[col.key] > 85 ? "bg-green-500" : (row as any)[col.key] > 60 ? "bg-blue-500" : "bg-yellow-500")}
                                                    style={{ width: `${(row as any)[col.key]}%` }}
                                                 ></div>
                                                 <span className="text-xs ml-1">{(row as any)[col.key]}</span>
                                             </div>
                                         ) : col.key === 'Verdict' ? (
                                             <span className={clsx(
                                                 "px-2 inline-flex text-xs leading-5 font-semibold rounded-full",
                                                 row.Verdict.includes("HIGH") ? "bg-green-100 text-green-800" :
                                                 row.Verdict.includes("PASS") ? "bg-blue-100 text-blue-800" :
                                                 row.Verdict.includes("WATCH") ? "bg-yellow-100 text-yellow-800" :
                                                 "bg-red-100 text-red-800"
                                             )}>
                                                 {row.Verdict}
                                             </span>
                                         ) : (row as any)[col.key]}
                                     </td>
                                 ))}
                             </tr>
                         ))}
                     </tbody>
                 </table>
             </div>
          </div>
      )}
    </div>
  );
};

export default ScanDashboard;
