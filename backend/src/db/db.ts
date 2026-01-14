import sqlite3 from 'sqlite3';
import path from 'path';

const DB_PATH = path.resolve(__dirname, '../../../fortress_history.db');

export const db = new sqlite3.Database(DB_PATH, (err) => {
    if (err) {
        console.error('Error opening database', err);
    } else {
        console.log('Database connected');
        initDb();
    }
});

function initDb() {
    db.serialize(() => {
        db.run(`CREATE TABLE IF NOT EXISTS scan_results
                (timestamp TEXT, symbol TEXT, score REAL, price REAL, verdict TEXT)`);

        db.run(`CREATE TABLE IF NOT EXISTS audit_logs
                (timestamp TEXT, action TEXT, universe TEXT, details TEXT)`);

        db.run(`CREATE TABLE IF NOT EXISTS smart_alerts
                (symbol TEXT, condition TEXT, status TEXT)`);
    });
}

export function getTableForUniverse(universe: string): string {
    if ("Nifty 50" === universe) return "scan_nifty50";
    if ("Nifty Next 50" === universe) return "scan_niftynext50";
    if (universe.includes("Nifty Midcap")) return "scan_midcap";
    if ("Nifty Midcap 150" === universe) return "scan_midcap";
    return "scan_results";
}

export function ensureTableExists(tableName: string): Promise<void> {
    return new Promise((resolve, reject) => {
        // Basic schema with core columns. Additional columns will be added dynamically if possible,
        // but for now we define the core schema to match Python's init or ensuring it acts like a relaxed schema.
        // Python code: checks if table exists, if not creates it. If exists, adds missing columns.
        // We will replicate a basic create here.
        // The columns are: timestamp, Symbol, Score, Price, Verdict, RSI, News, Events, Sector, Position_Qty, Stop_Loss, Target_10D, Analysts, Tgt_High, Tgt_Median, Tgt_Low, Tgt_Mean, Dispersion_Alert, Ret_30D, Ret_60D, Ret_90D, Universe

        const createQuery = `CREATE TABLE IF NOT EXISTS ${tableName} (
            timestamp TEXT,
            Symbol TEXT,
            Score REAL,
            Price REAL,
            Verdict TEXT,
            RSI REAL,
            News TEXT,
            Events TEXT,
            Sector TEXT,
            Position_Qty INTEGER,
            Stop_Loss REAL,
            Target_10D REAL,
            Analysts INTEGER,
            Tgt_High REAL,
            Tgt_Median REAL,
            Tgt_Low REAL,
            Tgt_Mean REAL,
            Dispersion_Alert TEXT,
            Ret_30D REAL,
            Ret_60D REAL,
            Ret_90D REAL,
            Universe TEXT
        )`;

        db.run(createQuery, (err) => {
            if (err) reject(err);
            else resolve();
        });
    });
}
