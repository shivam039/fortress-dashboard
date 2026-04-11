import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = process.env.NODE_ENV === 'production'
  ? 'https://shivam039-dev-fortress-engine.hf.space'
  : 'http://localhost:8000';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const universe = searchParams.get('universe') || 'Nifty 50';
    
    console.log(`[SectorPulse] Fetching from: ${BACKEND_URL}/api/sector-pulse?universe=${universe}`);

    const response = await fetch(`${BACKEND_URL}/api/sector-pulse?universe=${encodeURIComponent(universe)}`, {
      timeout: 10000,
      headers: {
        'Content-Type': 'application/json',
      },
    });
    
    console.log(`[SectorPulse] Response status: ${response.status}`);
    
    if (!response.ok) {
      console.error(`[SectorPulse] Backend returned status ${response.status}`);
      return NextResponse.json(
        { error: `Backend error: ${response.status}` },
        { status: 502 }
      );
    }
    
    const data = await response.json();
    console.log(`[SectorPulse] Success for universe: ${universe}`);
    return NextResponse.json(data);
  } catch (error) {
    console.error(`[SectorPulse] Error:`, error instanceof Error ? error.message : String(error));
    return NextResponse.json(
      { error: 'Backend unavailable', details: error instanceof Error ? error.message : String(error) },
      { status: 502 }
    );
  }
}