import { NextRequest, NextResponse } from 'next/server';
import { fetchWithRetry, BACKEND_URL } from '@/lib/api';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const limit = searchParams.get('limit');
    
    const endpoint = limit 
      ? `/api/mf-analysis?limit=${limit}` 
      : `/api/mf-analysis`;
    const url = `${BACKEND_URL}${endpoint}`;

    console.log(`[API Proxy] MF Analysis GET: ${url}`);

    const response = await fetchWithRetry(url, {
      headers: {
        'Accept': 'application/json',
      },
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`[API Proxy] Backend error (${response.status}): ${errorText}`);
      return NextResponse.json(
        { error: 'Backend error', status: response.status, details: errorText },
        { status: 502 }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error: any) {
    console.error(`[API Proxy] Fatal error fetching MF analysis:`, error.message || error);
    return NextResponse.json(
      { 
        error: 'Backend unavailable', 
        details: error.message || 'Network error or timeout',
        hint: 'Check if the Hugging Face space is sleeping or down.'
      },
      { status: 503 }
    );
  }
}