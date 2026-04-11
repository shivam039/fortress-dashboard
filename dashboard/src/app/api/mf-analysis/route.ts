import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = process.env.NODE_ENV === 'production'
  ? 'https://shivam039-dev-fortress-engine.hf.space'
  : 'http://localhost:8000';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const limit = searchParams.get('limit');

    const url = limit
      ? `${BACKEND_URL}/api/mf-analysis?limit=${limit}`
      : `${BACKEND_URL}/api/mf-analysis`;
    
    console.log(`[MFAnalysis] Fetching from: ${url}`);

    const response = await fetch(url, {
      timeout: 10000,
      headers: {
        'Content-Type': 'application/json',
      },
    });
    
    console.log(`[MFAnalysis] Response status: ${response.status}`);
    
    if (!response.ok) {
      console.error(`[MFAnalysis] Backend returned status ${response.status}`);
      return NextResponse.json(
        { error: `Backend error: ${response.status}` },
        { status: 502 }
      );
    }
    
    const data = await response.json();
    console.log(`[MFAnalysis] Success`);
    return NextResponse.json(data);
  } catch (error) {
    console.error(`[MFAnalysis] Error:`, error instanceof Error ? error.message : String(error));
    return NextResponse.json(
      { error: 'Backend unavailable', details: error instanceof Error ? error.message : String(error) },
      { status: 502 }
    );
  }
}