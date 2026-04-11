import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = process.env.NODE_ENV === 'production'
  ? 'https://shivam039-dev-fortress-engine.hf.space'
  : 'http://localhost:8000';

export async function GET() {
  try {
    console.log(`[Commodities] Fetching from: ${BACKEND_URL}/api/commodities`);
    
    const response = await fetch(`${BACKEND_URL}/api/commodities`, {
      timeout: 10000,
      headers: {
        'Content-Type': 'application/json',
      },
    });
    
    console.log(`[Commodities] Response status: ${response.status}`);
    
    if (!response.ok) {
      console.error(`[Commodities] Backend returned status ${response.status}`);
      return NextResponse.json(
        { error: `Backend error: ${response.status}` },
        { status: 502 }
      );
    }
    
    const data = await response.json();
    console.log(`[Commodities] Success`);
    return NextResponse.json(data);
  } catch (error) {
    console.error(`[Commodities] Error:`, error instanceof Error ? error.message : String(error));
    return NextResponse.json(
      { error: 'Backend unavailable', details: error instanceof Error ? error.message : String(error) },
      { status: 502 }
    );
  }
}