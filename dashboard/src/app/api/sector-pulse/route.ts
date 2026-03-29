import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = process.env.NODE_ENV === 'production'
  ? 'https://shivam039-dev-fortress-engine.hf.space'
  : 'http://localhost:8000';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const universe = searchParams.get('universe') || 'Nifty 50';

    const response = await fetch(`${BACKEND_URL}/api/sector-pulse?universe=${encodeURIComponent(universe)}`);
    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json({ error: 'Backend unavailable' }, { status: 500 });
  }
}