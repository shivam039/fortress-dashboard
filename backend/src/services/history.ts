import { db, getTableForUniverse } from '../db/db';

export function getSectorAnalysis(universe: string): Promise<any[]> {
    return new Promise((resolve, reject) => {
        const table = getTableForUniverse(universe);
        // We want average score change for sectors based on latest two timestamps?
        // Or just current average score?
        // The requirement says "Highlight top performing sectors based on conviction trend".
        // Let's get the latest scan results and average score by sector.

        db.all(`SELECT timestamp FROM ${table} ORDER BY timestamp DESC LIMIT 1`, (err, rows: any[]) => {
            if (err) {
                 if (err.message.includes('no such table')) resolve([]);
                 else reject(err);
                 return;
            }
            if (rows.length === 0) {
                resolve([]);
                return;
            }
            const latestTs = rows[0].timestamp;

            db.all(`SELECT Sector, AVG(Score) as AvgScore, COUNT(*) as Count FROM ${table} WHERE timestamp = ? GROUP BY Sector ORDER BY AvgScore DESC`,
                [latestTs], (err2, secRows) => {
                if (err2) reject(err2);
                else resolve(secRows);
            });
        });
    });
}

export function getTickerHistory(universe: string, symbol: string): Promise<any[]> {
    return new Promise((resolve, reject) => {
        const table = getTableForUniverse(universe);
        db.all(`SELECT * FROM ${table} WHERE Symbol = ? ORDER BY timestamp DESC`, [symbol], (err, rows) => {
            if (err) reject(err);
            else resolve(rows);
        });
    });
}
