import axios from 'axios';
import { io } from 'socket.io-client';

const API_URL = 'http://localhost:3001/api';
const SOCKET_URL = 'http://localhost:3001';

export const api = axios.create({
  baseURL: API_URL,
});

export const socket = io(SOCKET_URL);

export interface ScanResult {
    Symbol: string;
    Verdict: string;
    Score: number;
    Price: number;
    RSI: number;
    News: string;
    Events: string;
    Sector: string;
    Position_Qty: number;
    Stop_Loss: number;
    Target_10D: number;
    Analysts: number;
    Tgt_High: number;
    Tgt_Median: number;
    Tgt_Low: number;
    Tgt_Mean: number;
    Dispersion_Alert: string;
    Ret_30D: number | null;
    Ret_60D: number | null;
    Ret_90D: number | null;
    Universe: string;
}

export const fetchUniverses = async () => {
    const res = await api.get<string[]>('/universes');
    return res.data;
};

export const startScan = async (universe: string, portfolioVal: number, riskPct: number) => {
    const res = await api.post('/scan', { universe, portfolioVal, riskPct });
    return res.data;
};

export const fetchTimestamps = async (universe: string) => {
    const res = await api.get<string[]>('/history/timestamps', { params: { universe } });
    return res.data;
};

export const fetchHistoryResults = async (universe: string, timestamp: string) => {
    const res = await api.get<ScanResult[]>('/history/results', { params: { universe, timestamp } });
    return res.data;
};

export const compareScans = async (universe: string, tNew: string, tOld: string) => {
    const res = await api.get<any[]>('/history/compare', { params: { universe, tNew, tOld } });
    return res.data;
};

export const fetchAuditLogs = async () => {
    const res = await api.get<any[]>('/audit-logs');
    return res.data;
};

export const rollbackScan = async (universe: string, timestamp: string) => {
    const res = await api.post('/history/rollback', { universe, timestamp });
    return res.data;
};

export const fetchTickerHistory = async (universe: string, symbol: string) => {
    const res = await api.get<any[]>('/ticker/history', { params: { universe, symbol } });
    return res.data;
};

export const fetchSectorAnalysis = async (universe: string) => {
    const res = await api.get<any[]>('/sector/analysis', { params: { universe } });
    return res.data;
};
