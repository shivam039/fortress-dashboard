import yahooFinance from 'yahoo-finance2';
import { calculateEMA, calculateRSI, calculateATR, calculateSupertrend } from './indicators';
import { db, ensureTableExists, getTableForUniverse } from '../db/db';
import { TICKER_GROUPS, SECTOR_MAP } from '../config';
import { DateTime } from 'luxon';

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

export async function fetchAndScanTicker(ticker: string, universe: string, portfolioVal: number, riskPct: number): Promise<ScanResult | null> {
    try {
        // Fetch 2 years of data
        const today = new Date();
        const twoYearsAgo = new Date();
        twoYearsAgo.setFullYear(today.getFullYear() - 2);

        const queryOptions = { period1: twoYearsAgo, period2: today }; // yahoo-finance2 format
        const historical = await yahooFinance.historical(ticker, queryOptions) as any[];
        const quote = await yahooFinance.quote(ticker);
        const search = await yahooFinance.search(ticker, { newsCount: 5 }) as any;

        if (!historical || historical.length < 210) return null;

        const close = historical.map((d: any) => d.close);
        const high = historical.map((d: any) => d.high);
        const low = historical.map((d: any) => d.low);

        // Indicators
        const ema200Arr = calculateEMA(close, 200);
        const rsiArr = calculateRSI(close, 14);
        const atrArr = calculateATR(high, low, close, 14);
        const stArr = calculateSupertrend(high, low, close, 10, 3);

        if (ema200Arr.length === 0 || rsiArr.length === 0 || atrArr.length === 0 || stArr.length === 0) return null;

        const price = close[close.length - 1];
        const ema200 = ema200Arr[ema200Arr.length - 1];
        const rsi = rsiArr[rsiArr.length - 1];
        const atr = atrArr[atrArr.length - 1];
        const supertrend = stArr[stArr.length - 1];
        const trendDir = supertrend.direction; // 1 or -1

        const techBase = price > ema200 && trendDir === 1;

        const slDistance = atr * 1.5;
        const slPrice = parseFloat((price - slDistance).toFixed(2));
        const target10d = parseFloat((price + atr * 1.8).toFixed(2));

        const riskAmount = portfolioVal * riskPct;
        const posSize = slDistance > 0 ? Math.floor(riskAmount / slDistance) : 0;

        // News & Events
        let scoreMod = 0;
        let newsSentiment = "Neutral";
        let eventStatus = "✅ Safe";

        if (search && search.news) {
            const newsTitles = search.news.map((n: any) => n.title.toLowerCase()).join(' ');
            if (['fraud', 'investigation', 'default', 'bankruptcy', 'scam', 'legal'].some((k: string) => newsTitles.includes(k))) {
                newsSentiment = "🚨 BLACK SWAN";
                scoreMod -= 40;
            }
        }

        // Yahoo Finance 2 quote might have earnings info or we need to fetch calendar?
        // quoteSummary is separate call. Let's assume simple check or skip if complex.
        // We can get earnings date from quoteSummary if needed, but quote often has it.
        // quote.earningsTimestamp

        // Check institutional fortress logic
        let conviction = 0;

        // Analysts
        // We need quoteSummary for analyst data usually.
        // `quote` usually has some info but `quoteSummary` is richer.
        // We'll try to get what we can from `quote`.
        // yahoo-finance2 `quote` result:
        /*
         averageAnalystRating: '2.1 - Buy',
         */
        // But numeric targets might be in quoteSummary.
        // Let's do a quick separate call if needed or just use 0 defaults to be safe.
        // Actually, let's try to get quoteSummary for 'financialData' and 'defaultKeyStatistics'.

        let analystCount = 0;
        let targetHigh = 0;
        let targetLow = 0;
        let targetMedian = 0;
        let targetMean = 0;

        // Optimization: Use a single call if possible. quoteSummary(ticker, { modules: [...] })
        // But for speed in loop, maybe we stick to simple quote or handle error gracefully.
        // The original code used `ticker_obj.info` which implies yfinance full info fetch.
        // That is slow.

        try {
           const summary: any = await yahooFinance.quoteSummary(ticker, { modules: ['financialData', 'defaultKeyStatistics'] });
           if (summary.financialData) {
               analystCount = summary.financialData.numberOfAnalystOpinions || 0;
               targetHigh = summary.financialData.targetHighPrice || 0;
               targetLow = summary.financialData.targetLowPrice || 0;
               targetMedian = summary.financialData.targetMedianPrice || 0;
               targetMean = summary.financialData.targetMeanPrice || 0;
           }
        } catch (e) {
            // ignore
        }

        if (techBase) {
            conviction += 60;
            if (rsi >= 48 && rsi <= 62) conviction += 20;
            else if (rsi >= 40 && rsi <= 72) conviction += 10;
            conviction += scoreMod;
        }

        const dispersionPct = price > 0 ? ((targetHigh - targetLow) / price) * 100 : 0;
        const dispersionAlert = dispersionPct > 30 ? "⚠️ High Dispersion" : "✅";
        if (dispersionPct > 30) conviction -= 10;

        conviction = Math.max(0, Math.min(100, conviction));

        let verdict = "❌ FAIL";
        if (conviction >= 85) verdict = "🔥 HIGH";
        else if (conviction >= 60) verdict = "🚀 PASS";
        else if (techBase) verdict = "🟡 WATCH";

        // Returns
        const returns: any = {};
        const currentDate = historical[historical.length-1].date; // Date object

        [30, 60, 90].forEach(days => {
            const targetTime = currentDate.getTime() - (days * 24 * 60 * 60 * 1000);
            // Find closest date in history
            // History is ordered by date? Yes usually.
            // efficient search or find
            const past = historical.find((d: any) => d.date.getTime() >= targetTime); // simple approx since sorted ascending
            if (past) {
                const pastPrice = past.close;
                const pctChange = ((price - pastPrice) / pastPrice) * 100;
                returns[`Ret_${days}D`] = parseFloat(pctChange.toFixed(2));
            } else {
                returns[`Ret_${days}D`] = null;
            }
        });

        return {
            Symbol: ticker,
            Verdict: verdict,
            Score: conviction,
            Price: parseFloat(price.toFixed(2)),
            RSI: parseFloat(rsi.toFixed(1)),
            News: newsSentiment,
            Events: eventStatus,
            Sector: SECTOR_MAP[ticker] || "General",
            Position_Qty: posSize,
            Stop_Loss: slPrice,
            Target_10D: target10d,
            Analysts: analystCount,
            Tgt_High: targetHigh,
            Tgt_Median: targetMedian,
            Tgt_Low: targetLow,
            Tgt_Mean: targetMean,
            Dispersion_Alert: dispersionAlert,
            Ret_30D: returns['Ret_30D'],
            Ret_60D: returns['Ret_60D'],
            Ret_90D: returns['Ret_90D'],
            Universe: universe
        };

    } catch (error) {
        console.error(`Error scanning ${ticker}:`, error);
        return null;
    }
}

export async function runScanForUniverse(universe: string, portfolioVal: number, riskPct: number, updateCallback: (progress: number, message: string, result?: ScanResult) => void) {
    const tickers = TICKER_GROUPS[universe];
    if (!tickers) throw new Error("Invalid Universe");

    const results: ScanResult[] = [];
    const tableName = getTableForUniverse(universe);
    await ensureTableExists(tableName);

    for (let i = 0; i < tickers.length; i++) {
        const ticker = tickers[i];
        const progress = ((i + 1) / tickers.length) * 100;
        updateCallback(progress, `Scanning ${ticker} (${i + 1}/${tickers.length})`);

        // Add delay to avoid rate limits
        await new Promise(r => setTimeout(r, 700));

        const res = await fetchAndScanTicker(ticker, universe, portfolioVal, riskPct);
        if (res) {
            results.push(res);
            updateCallback(progress, `Scanned ${ticker}`, res);
        }
    }

    // Save results to DB
    // We append to the table.
    // timestamp is generated now.
    const timestamp = DateTime.now().toFormat('yyyy-MM-dd HH:mm:ss');
    const stmt = db.prepare(`INSERT INTO ${tableName} (
        timestamp, Symbol, Score, Price, Verdict, RSI, News, Events, Sector, Position_Qty, Stop_Loss,
        Target_10D, Analysts, Tgt_High, Tgt_Median, Tgt_Low, Tgt_Mean, Dispersion_Alert,
        Ret_30D, Ret_60D, Ret_90D, Universe
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`);

    results.forEach(r => {
        stmt.run(
            timestamp, r.Symbol, r.Score, r.Price, r.Verdict, r.RSI, r.News, r.Events, r.Sector, r.Position_Qty, r.Stop_Loss,
            r.Target_10D, r.Analysts, r.Tgt_High, r.Tgt_Median, r.Tgt_Low, r.Tgt_Mean, r.Dispersion_Alert,
            r.Ret_30D, r.Ret_60D, r.Ret_90D, r.Universe
        );
    });
    stmt.finalize();

    // Log Audit
    db.run("INSERT INTO audit_logs (timestamp, action, universe, details) VALUES (?, ?, ?, ?)",
        [timestamp, "Scan Completed", universe, `Saved ${results.length} records to ${tableName}`]);

    return { results, timestamp };
}
