import { SMA, EMA, RSI, ATR } from 'technicalindicators';

// Helper to handle data series
export function calculateEMA(values: number[], period: number): number[] {
    return EMA.calculate({ period, values });
}

export function calculateRSI(values: number[], period: number = 14): number[] {
    return RSI.calculate({ period, values });
}

export function calculateATR(high: number[], low: number[], close: number[], period: number = 14): number[] {
    return ATR.calculate({ period, high, low, close });
}

export interface SupertrendResult {
    upperBand: number;
    lowerBand: number;
    superTrend: number; // The value of the line
    direction: 1 | -1; // 1 for Bullish, -1 for Bearish
}

// Custom Supertrend implementation to match pandas_ta defaults (10, 3)
// pandas_ta implementation is basically:
// basicUpper = (high + low) / 2 + multiplier * atr
// basicLower = (high + low) / 2 - multiplier * atr
// if close > prevUpper then true ...
export function calculateSupertrend(
    high: number[],
    low: number[],
    close: number[],
    period: number = 10,
    multiplier: number = 3
): SupertrendResult[] {
    const atr = calculateATR(high, low, close, period);
    const results: SupertrendResult[] = [];

    // ATR result length is shorter than input by period-1 usually, but technicalindicators library might behave differently.
    // technicalindicators ATR result is length = input_length - period + 1 (roughly? no, first period elements are used for initial calculation)
    // Actually technicalindicators returns result starting from the point it can calculate.
    // We need to align indices.

    // Let's assume we align from the end.

    let trend = 1; // Default trend
    let lowerBand = 0;
    let upperBand = 0;

    // We need to iterate and calculate.
    // The first ATR value corresponds to index `period` (0-based) in the input arrays?
    // Let's implement a manual Supertrend loop to be safe and precise.
    // Or we use the ATR from the library and map it back.

    // ATR from library:
    // If input has 100 elements, period 14.
    // Output has 87 elements? (100 - 14 + 1?)
    // Let's verify loop.

    // For simplicity, let's implement ATR manually inside or just use the library and pad/align.
    // The library ATR[0] corresponds to the ATR at index (period-1) or period?

    // Aligning:
    const diff = close.length - atr.length;

    // We can only calculate Supertrend where we have ATR.
    // So we start loop from `diff`.

    // Initialize previous bands
    let prevUpper = 0;
    let prevLower = 0;
    let prevTrend = 1;

    for (let i = 0; i < atr.length; i++) {
        const idx = i + diff;
        const currHigh = high[idx];
        const currLow = low[idx];
        const currClose = close[idx];
        const currAtr = atr[i];

        const basicUpper = (currHigh + currLow) / 2 + multiplier * currAtr;
        const basicLower = (currHigh + currLow) / 2 - multiplier * currAtr;

        let currUpper = basicUpper;
        let currLower = basicLower;

        if (i > 0) {
            // Logic from pandas_ta / typical supertrend
            // if prevClose <= prevUpper: currUpper = min(basicUpper, prevUpper)
            const prevClose = close[idx-1];

            if (basicUpper < prevUpper || prevClose > prevUpper) {
                currUpper = basicUpper;
            } else {
                currUpper = prevUpper;
            }

            if (basicLower > prevLower || prevClose < prevLower) {
                currLower = basicLower;
            } else {
                currLower = prevLower;
            }
        }

        let currTrend = prevTrend;
        if (prevTrend === 1) {
            if (currClose < currLower) {
                currTrend = -1;
            }
        } else {
            if (currClose > currUpper) {
                currTrend = 1;
            }
        }

        results.push({
            upperBand: currUpper,
            lowerBand: currLower,
            superTrend: currTrend === 1 ? currLower : currUpper,
            direction: currTrend === 1 ? 1 : -1
        });

        prevUpper = currUpper;
        prevLower = currLower;
        prevTrend = currTrend;
    }

    return results;
}
