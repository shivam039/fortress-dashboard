import { NextRequest, NextResponse } from 'next/server';
import { fetchWithRetry, BACKEND_URL } from '@/lib/api';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const url = `${BACKEND_URL}/api/scan`;

    console.log(`[API Proxy] Scan POST: ${url}`);

    const response = await fetchWithRetry(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`[API Proxy] Scan backend error (${response.status}): ${errorText}`);
      return NextResponse.json(
        { error: 'Backend error', status: response.status, details: errorText },
        { status: 502 }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error: any) {
    console.error(`[API Proxy] Fatal error during scan:`, error.message || error);
    return NextResponse.json(
      { error: 'Backend unavailable', details: error.message },
      { status: 503 }
    );
  }
}