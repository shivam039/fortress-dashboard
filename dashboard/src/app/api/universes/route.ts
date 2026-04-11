import { NextRequest, NextResponse } from 'next/server';
import { fetchWithRetry, BACKEND_URL } from '@/lib/api';

export async function GET() {
  try {
    const url = `${BACKEND_URL}/api/universes`;
    console.log(`[API Proxy] Universes GET: ${url}`);

    const response = await fetchWithRetry(url, {
      headers: {
        'Accept': 'application/json',
      },
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`[API Proxy] Universes backend error (${response.status}): ${errorText}`);
      return NextResponse.json(
        { error: 'Backend error', status: response.status },
        { status: 502 }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error: any) {
    console.error(`[API Proxy] Fatal error fetching universes:`, error.message || error);
    return NextResponse.json(
      { error: 'Backend unavailable', details: error.message },
      { status: 503 }
    );
  }
}