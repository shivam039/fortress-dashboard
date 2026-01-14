import React, { useState, useEffect } from 'react';
import { fetchTimestamps, compareScans, fetchUniverses, fetchSectorAnalysis } from '../services/api';
import { ArrowUp, ArrowDown, Download } from 'lucide-react';
import clsx from 'clsx';
import TrendChart from './TrendChart';
import jsPDF from 'jspdf';
import autoTable from 'jspdf-autotable';

const ScanHistory: React.FC = () => {
    const [universes, setUniverses] = useState<string[]>([]);
    const [selectedUniverse, setSelectedUniverse] = useState('');
    const [timestamps, setTimestamps] = useState<string[]>([]);
    const [tNew, setTNew] = useState('');
    const [tOld, setTOld] = useState('');
    const [comparison, setComparison] = useState<any[]>([]);
    const [sectors, setSectors] = useState<any[]>([]);

    useEffect(() => {
        fetchUniverses().then(data => {
            setUniverses(data);
            if(data.length > 0) setSelectedUniverse(data[0]);
        });
    }, []);

    useEffect(() => {
        if (selectedUniverse) {
            fetchTimestamps(selectedUniverse).then(ts => {
                setTimestamps(ts);
                if (ts.length > 0) setTNew(ts[0]);
                if (ts.length > 1) setTOld(ts[1]);
            });
            fetchSectorAnalysis(selectedUniverse).then(setSectors);
        }
    }, [selectedUniverse]);

    const handleCompare = async () => {
        if (selectedUniverse && tNew && tOld) {
            const data = await compareScans(selectedUniverse, tNew, tOld);
            setComparison(data);
        }
    };

    const exportCSV = () => {
         if (comparison.length === 0) return;
         const headers = Object.keys(comparison[0]).join(',');
         const rows = comparison.map(r => Object.values(r).join(',')).join('\n');
         const csv = `${headers}\n${rows}`;
         const blob = new Blob([csv], { type: 'text/csv' });
         const url = window.URL.createObjectURL(blob);
         const a = document.createElement('a');
         a.href = url;
         a.download = `Comparison_${tNew}_vs_${tOld}.csv`;
         a.click();
    };

    const exportPDF = () => {
        if (comparison.length === 0) return;
        const doc = new jsPDF();

        doc.setFontSize(16);
        doc.text(`Fortress Scan Comparison`, 14, 15);
        doc.setFontSize(10);
        doc.text(`${tNew} vs ${tOld}`, 14, 22);

        // Summary
        const newC = comparison.filter(c => c.Verdict_Shift === "New").length;
        const dropC = comparison.filter(c => c.Verdict_Shift === "Dropped").length;
        const impC = comparison.filter(c => c.Score_Change > 0).length;
        const decC = comparison.filter(c => c.Score_Change < 0).length;

        doc.text(`New: ${newC} | Dropped: ${dropC} | Improvers: ${impC} | Decliners: ${decC}`, 14, 30);

        autoTable(doc, {
            startY: 35,
            head: [['Symbol', 'Score (New)', 'Score (Old)', 'Delta', 'Verdict', 'Price Chg']],
            body: comparison.map(r => [
                r.Symbol,
                r.Score_new,
                r.Score_old,
                r.Score_Change > 0 ? `+${r.Score_Change}` : r.Score_Change,
                r.Verdict_Shift,
                `${r.Price_Chg_Pct.toFixed(2)}%`
            ]),
        });

        doc.save(`Scan_Report_${tNew}.pdf`);
    };

    const newEntrants = comparison.filter(c => c.Verdict_Shift === "New");
    const dropped = comparison.filter(c => c.Verdict_Shift === "Dropped");
    const improvers = comparison.filter(c => c.Score_Change > 0);
    const decliners = comparison.filter(c => c.Score_Change < 0);

    return (
        <div className="p-6 space-y-6">
            <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
                <h2 className="text-xl font-bold mb-4 text-slate-800">📜 Scan History Intelligence</h2>

                <div className="grid grid-cols-1 md:grid-cols-4 gap-6 items-end">
                    <div>
                        <label className="block text-sm font-medium text-slate-600 mb-1">Universe</label>
                        <select
                            className="w-full border rounded-lg p-2 bg-slate-50"
                            value={selectedUniverse}
                            onChange={e => setSelectedUniverse(e.target.value)}
                        >
                            {universes.map(u => <option key={u} value={u}>{u}</option>)}
                        </select>
                    </div>
                    <div>
                         <label className="block text-sm font-medium text-slate-600 mb-1">New Scan</label>
                         <select
                            className="w-full border rounded-lg p-2 bg-slate-50"
                            value={tNew}
                            onChange={e => setTNew(e.target.value)}
                         >
                             {timestamps.map(t => <option key={t} value={t}>{t}</option>)}
                         </select>
                    </div>
                    <div>
                         <label className="block text-sm font-medium text-slate-600 mb-1">Old Scan</label>
                         <select
                            className="w-full border rounded-lg p-2 bg-slate-50"
                            value={tOld}
                            onChange={e => setTOld(e.target.value)}
                         >
                             {timestamps.map(t => <option key={t} value={t}>{t}</option>)}
                         </select>
                    </div>
                    <div>
                        <button
                           className="w-full py-2 px-4 rounded-lg font-bold text-white bg-indigo-600 hover:bg-indigo-700"
                           onClick={handleCompare}
                        >
                           Compare Scans
                        </button>
                    </div>
                </div>
            </div>

            {sectors.length > 0 && (
                 <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
                    <h3 className="text-lg font-bold mb-4">🧠 AI Sector Intelligence</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div>
                             <h4 className="font-semibold text-green-700 mb-2">Top Performing Sectors</h4>
                             {sectors.slice(0, 3).map((s, i) => (
                                 <div key={i} className="flex justify-between border-b py-2 last:border-0">
                                     <span>{s.Sector}</span>
                                     <span className="font-bold">{s.AvgScore.toFixed(1)}</span>
                                 </div>
                             ))}
                        </div>
                        <div>
                             <h4 className="font-semibold text-red-700 mb-2">Lagging Sectors</h4>
                             {[...sectors].reverse().slice(0, 3).map((s, i) => (
                                 <div key={i} className="flex justify-between border-b py-2 last:border-0">
                                     <span>{s.Sector}</span>
                                     <span className="font-bold">{s.AvgScore.toFixed(1)}</span>
                                 </div>
                             ))}
                        </div>
                    </div>
                 </div>
            )}

            {comparison.length > 0 && (
                <>
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                         <div className="bg-white p-4 rounded-lg shadow-sm border border-slate-200">
                             <div className="text-sm text-slate-500">New Entrants</div>
                             <div className="text-2xl font-bold text-green-600">+{newEntrants.length}</div>
                         </div>
                         <div className="bg-white p-4 rounded-lg shadow-sm border border-slate-200">
                             <div className="text-sm text-slate-500">Dropped</div>
                             <div className="text-2xl font-bold text-red-600">-{dropped.length}</div>
                         </div>
                         <div className="bg-white p-4 rounded-lg shadow-sm border border-slate-200">
                             <div className="text-sm text-slate-500">Improvers</div>
                             <div className="text-2xl font-bold text-blue-600">{improvers.length}</div>
                         </div>
                         <div className="bg-white p-4 rounded-lg shadow-sm border border-slate-200">
                             <div className="text-sm text-slate-500">Decliners</div>
                             <div className="text-2xl font-bold text-orange-600">{decliners.length}</div>
                         </div>
                    </div>

                    <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
                        <div className="flex justify-between items-center mb-4">
                            <h3 className="text-lg font-bold">Comparison Table</h3>
                            <div className="flex gap-2">
                                <button onClick={exportCSV} className="flex items-center gap-2 text-indigo-600 hover:text-indigo-800 bg-indigo-50 px-3 py-1 rounded">
                                    <Download size={18} /> CSV
                                </button>
                                <button onClick={exportPDF} className="flex items-center gap-2 text-red-600 hover:text-red-800 bg-red-50 px-3 py-1 rounded">
                                    <Download size={18} /> PDF
                                </button>
                            </div>
                        </div>
                        <div className="overflow-x-auto">
                            <table className="min-w-full divide-y divide-slate-200">
                                <thead className="bg-slate-50">
                                    <tr>
                                        <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase">Symbol</th>
                                        <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase">Score (New)</th>
                                        <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase">Score (Old)</th>
                                        <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase">Delta</th>
                                        <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase">Verdict Shift</th>
                                        <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase">Price Change</th>
                                    </tr>
                                </thead>
                                <tbody className="bg-white divide-y divide-slate-200">
                                    {comparison.sort((a,b) => b.Score_Change - a.Score_Change).map((row, idx) => (
                                        <tr key={idx} className={idx % 2 === 0 ? 'bg-white' : 'bg-slate-50'}>
                                            <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-slate-900">{row.Symbol}</td>
                                            <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-700">{row.Score_new}</td>
                                            <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-700">{row.Score_old}</td>
                                            <td className={clsx("px-6 py-4 whitespace-nowrap text-sm font-bold",
                                                row.Score_Change > 0 ? "text-green-600" : row.Score_Change < 0 ? "text-red-600" : "text-slate-500"
                                            )}>
                                                {row.Score_Change > 0 ? `+${row.Score_Change}` : row.Score_Change}
                                            </td>
                                            <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-700">{row.Verdict_Shift}</td>
                                            <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-700">
                                                {row.Price_Change.toFixed(2)} ({row.Price_Chg_Pct.toFixed(2)}%)
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
                            <h3 className="text-lg font-bold mb-4">🚀 Top Gainers</h3>
                            {improvers.slice(0, 5).map(i => (
                                <div key={i.Symbol} className="flex justify-between items-center py-2 border-b last:border-0">
                                    <span>{i.Symbol}</span>
                                    <span className="text-green-600 font-bold">+{i.Score_Change}</span>
                                </div>
                            ))}
                        </div>
                        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
                            <h3 className="text-lg font-bold mb-4">📉 Top Losers</h3>
                            {decliners.slice(0, 5).map(i => (
                                <div key={i.Symbol} className="flex justify-between items-center py-2 border-b last:border-0">
                                    <span>{i.Symbol}</span>
                                    <span className="text-red-600 font-bold">{i.Score_Change}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                </>
            )}

            <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
                <h2 className="text-xl font-bold mb-4 text-slate-800">📈 Conviction Trend Engine</h2>
                <TrendChart universe={selectedUniverse} />
            </div>
        </div>
    );
};

export default ScanHistory;
