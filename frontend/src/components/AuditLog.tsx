import React, { useState, useEffect } from 'react';
import { fetchAuditLogs, rollbackScan, fetchUniverses, fetchTimestamps } from '../services/api';
import { AlertTriangle, RotateCcw } from 'lucide-react';
import clsx from 'clsx';

const AuditLog: React.FC = () => {
    const [logs, setLogs] = useState<any[]>([]);
    const [universes, setUniverses] = useState<string[]>([]);
    const [selectedUniverse, setSelectedUniverse] = useState('');
    const [timestamps, setTimestamps] = useState<string[]>([]);
    const [latestScan, setLatestScan] = useState('');

    const refreshLogs = () => {
        fetchAuditLogs().then(setLogs);
    };

    useEffect(() => {
        refreshLogs();
        fetchUniverses().then(data => {
            setUniverses(data);
            if (data.length > 0) setSelectedUniverse(data[0]);
        });
    }, []);

    useEffect(() => {
        if (selectedUniverse) {
            fetchTimestamps(selectedUniverse).then(ts => {
                setTimestamps(ts);
                if (ts.length > 0) setLatestScan(ts[0]);
                else setLatestScan('');
            });
        }
    }, [selectedUniverse]);

    const handleRollback = async () => {
        if (!selectedUniverse || !latestScan) return;
        if (confirm(`Are you sure you want to delete scan from ${latestScan}? This cannot be undone.`)) {
            await rollbackScan(selectedUniverse, latestScan);
            refreshLogs();
            // Refresh timestamps
            const ts = await fetchTimestamps(selectedUniverse);
            setTimestamps(ts);
            if (ts.length > 0) setLatestScan(ts[0]);
            else setLatestScan('');
        }
    };

    return (
        <div className="p-6 space-y-6">
            <h2 className="text-xl font-bold text-slate-800">🛡️ System Audit Logs & Rollback</h2>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="md:col-span-2 bg-white p-6 rounded-xl shadow-sm border border-slate-200">
                    <h3 className="text-lg font-bold mb-4">Activity Log</h3>
                    <div className="overflow-x-auto max-h-[500px]">
                        <table className="min-w-full divide-y divide-slate-200">
                            <thead className="bg-slate-50 sticky top-0">
                                <tr>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase">Timestamp</th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase">Action</th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase">Universe</th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase">Details</th>
                                </tr>
                            </thead>
                            <tbody className="bg-white divide-y divide-slate-200">
                                {logs.map((log, idx) => (
                                    <tr key={idx}>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500">{log.timestamp}</td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-slate-900">{log.action}</td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500">{log.universe}</td>
                                        <td className="px-6 py-4 text-sm text-slate-500">{log.details}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>

                <div className="bg-white p-6 rounded-xl shadow-sm border border-red-100">
                    <div className="flex items-center gap-2 mb-4 text-red-600">
                        <AlertTriangle size={24} />
                        <h3 className="text-lg font-bold">Danger Zone</h3>
                    </div>
                    <p className="text-sm text-slate-600 mb-4">
                        Rollback the latest scan for a universe if data is corrupted or scan was accidental.
                    </p>

                    <div className="space-y-4">
                        <div>
                            <label className="block text-sm font-medium text-slate-600 mb-1">Target Universe</label>
                            <select
                                className="w-full border rounded-lg p-2 bg-slate-50"
                                value={selectedUniverse}
                                onChange={e => setSelectedUniverse(e.target.value)}
                            >
                                {universes.map(u => <option key={u} value={u}>{u}</option>)}
                            </select>
                        </div>

                        {latestScan ? (
                            <div className="bg-slate-50 p-3 rounded border border-slate-200 text-sm">
                                <strong>Latest Scan:</strong> {latestScan}
                            </div>
                        ) : (
                            <div className="text-sm text-slate-400">No scans found.</div>
                        )}

                        <button
                            onClick={handleRollback}
                            disabled={!latestScan}
                            className={clsx(
                                "w-full py-2 px-4 rounded-lg font-bold text-white flex items-center justify-center gap-2",
                                latestScan ? "bg-red-600 hover:bg-red-700" : "bg-red-300 cursor-not-allowed"
                            )}
                        >
                            <RotateCcw size={18} /> Rollback Latest
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default AuditLog;
