import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = process.env.NODE_ENV === 'production'
  ? 'https://shivam039-dev-fortress-engine.hf.space'
  : 'http://localhost:8000';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    console.log(`[Scan] Posting to: ${BACKEND_URL}/api/scan`);
    
    const response = await fetch(`${BACKEND_URL}/api/scan`, {
      method: 'POST',
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });

    console.log(`[Scan] Response status: ${response.status}`);
    
    if (!response.ok) {
      console.error(`[Scan] Backend returned status ${response.status}`);
      throw new Error(`Backend responded with ${response.status}`);
    }

    const data = await response.json();
    console.log(`[Scan] Success`);
    return NextResponse.json(data);
  } catch (error) {
    console.error('[Scan] Error:', error instanceof Error ? error.message : String(error));
    return NextResponse.json(
      { error: 'Backend scan failed', details: error instanceof Error ? error.message : String(error) },
      { status: 502 }
    );
  }
}