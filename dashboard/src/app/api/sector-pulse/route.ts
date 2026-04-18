import { NextRequest, NextResponse } from 'next/server';
import { fetchWithRetry, BACKEND_URL } from '@/lib/api';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const universe = searchParams.get('universe') || 'Nifty 50';
    
    const url = `${BACKEND_URL}/api/sector-pulse?universe=${encodeURIComponent(universe)}`;

    console.log(`[API Proxy] Sector Pulse GET: ${url}`);

    const response = await fetchWithRetry(url, {
      headers: {
        'Accept': 'application/json',
      },
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`[API Proxy] Sector Pulse backend error (${response.status}): ${errorText}`);
      return NextResponse.json(
        { error: 'Backend error', status: response.status },
        { status: 502 }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error: any) {
    console.error(`[API Proxy] Fatal error fetching Sector Pulse:`, error.message || error);
    return NextResponse.json(
      { error: 'Backend unavailable', details: error.message },
      { status: 503 }
    );
  }
}