import React, { useState, useEffect } from 'react';
import { fetchTickerHistory } from '../services/api';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { TICKER_GROUPS } from '../../../backend/src/config'; // Can't import from backend directly in frontend usually.
// Wait, we can't import backend files. We need to fetch ticker list or use hardcoded/prop.
// Let's use an input or select that gets populated from API if possible, or we just type it.
// The history component already has universe selected.
import { fetchUniverses } from '../services/api';

// Since we cannot import from backend in vite (outside src), we rely on API.

interface TrendChartProps {
    universe: string;
}

const TrendChart: React.FC<TrendChartProps> = ({ universe }) => {
    const [symbol, setSymbol] = useState('');
    const [history, setHistory] = useState<any[]>([]);
    const [loading, setLoading] = useState(false);

    const handleSearch = async () => {
        if (!symbol || !universe) return;
        setLoading(true);
        try {
            const data = await fetchTickerHistory(universe, symbol);
            setHistory(data.map(d => ({...d, timestamp: d.timestamp.split(' ')[0]}))); // Simplify date
        } catch (e) {
            console.error(e);
        }
        setLoading(false);
    };

    return (
        <div>
            <div className="flex gap-4 mb-4">
                <input
                    type="text"
                    placeholder="Enter Symbol (e.g. RELIANCE.NS)"
                    className="border rounded-lg p-2 flex-1"
                    value={symbol}
                    onChange={e => setSymbol(e.target.value)}
                />
                <button
                    onClick={handleSearch}
                    className="bg-indigo-600 text-white px-4 py-2 rounded-lg hover:bg-indigo-700"
                    disabled={loading}
                >
                    {loading ? 'Loading...' : 'Analyze Trend'}
                </button>
            </div>

            {history.length > 0 ? (
                <div className="h-80 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={history}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="timestamp" />
                            <YAxis domain={[0, 100]} />
                            <Tooltip />
                            <Line type="monotone" dataKey="Score" stroke="#4f46e5" strokeWidth={2} activeDot={{ r: 8 }} />
                        </LineChart>
                    </ResponsiveContainer>
                    <div className="mt-4 flex gap-4 text-sm">
                        <div className="bg-slate-50 p-3 rounded border">
                            <span className="text-slate-500">Avg Score:</span>
                            <span className="font-bold ml-2">{(history.reduce((acc, curr) => acc + curr.Score, 0) / history.length).toFixed(1)}</span>
                        </div>
                        <div className="bg-slate-50 p-3 rounded border">
                            <span className="text-slate-500">Volatility:</span>
                            {/* Standard deviation calc */}
                            <span className="font-bold ml-2">
                                {(() => {
                                    const avg = history.reduce((acc, curr) => acc + curr.Score, 0) / history.length;
                                    const variance = history.reduce((acc, curr) => acc + Math.pow(curr.Score - avg, 2), 0) / history.length;
                                    return Math.sqrt(variance).toFixed(1);
                                })()}
                            </span>
                        </div>
                    </div>
                </div>
            ) : (
                <div className="text-center text-slate-400 py-10">
                    Enter a symbol to see historical conviction trend.
                </div>
            )}
        </div>
    );
};

export default TrendChart;
