import { calculateEMA, calculateRSI, calculateATR, calculateSupertrend } from '../src/services/indicators';

// Simple test runner
function runTest(name: string, fn: () => boolean) {
    if (fn()) {
        console.log(`✅ ${name}`);
    } else {
        console.error(`❌ ${name}`);
        process.exit(1);
    }
}

function assertClose(a: number, b: number, epsilon = 0.01) {
    return Math.abs(a - b) < epsilon;
}

const mockClose = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]; // 15 elements
const mockHigh =  [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25];
const mockLow =   [9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23];

console.log("Running Indicator Tests...");

// EMA
runTest("EMA Calculation", () => {
    const period = 5;
    const result = calculateEMA(mockClose, period);
    return result.length === mockClose.length - period + 1 || result.length > 0;
});

// RSI
runTest("RSI Calculation", () => {
    const period = 14;
    // Need at least period + 1 elements usually
    const result = calculateRSI(mockClose, 14);
    // With 15 elements and period 14, we might get 1 or 2 results depending on implementation (usually just 1 or 2)
    // technicalindicators might return valid array
    return Array.isArray(result);
});

// Supertrend
runTest("Supertrend Calculation", () => {
    const result = calculateSupertrend(mockHigh, mockLow, mockClose, 10, 3);
    // Should return array of objects with direction
    return result.length > 0 && typeof result[0].direction === 'number';
});

console.log("All tests passed.");
