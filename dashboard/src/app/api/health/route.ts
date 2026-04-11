import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = process.env.NODE_ENV === 'production'
  ? 'https://shivam039-dev-fortress-engine.hf.space'
  : 'http://localhost:8000';

async function fetchWithRetry(url: string, options: any, retries = 3) {
  let lastError: Error | null = null;
  
  for (let i = 0; i < retries; i++) {
    try {
      console.log(`[Health] Attempt ${i + 1}/${retries} to fetch ${url}`);
      const response = await fetch(url, {
        ...options,
        timeout: 10000,
      });
      return response;
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));
      console.warn(`[Health] Attempt ${i + 1} failed:`, lastError.message);
      if (i < retries - 1) {
        await new Promise(resolve => setTimeout(resolve, 500 * (i + 1)));
      }
    }
  }
  
  throw lastError;
}

export async function GET() {
  try {
    console.log(`[Health] Fetching from: ${BACKEND_URL}/api/health`);
    
    const response = await fetchWithRetry(`${BACKEND_URL}/api/health`, {
      headers: {
        'Content-Type': 'application/json',
      },
    });
    
    console.log(`[Health] Response status: ${response.status}`);
    
    if (!response.ok) {
      console.error(`[Health] Backend returned status ${response.status}`);
      return NextResponse.json(
        { error: `Backend error: ${response.status}` },
        { status: 502 }
      );
    }
    
    const data = await response.json();
    console.log(`[Health] Success:`, data);
    return NextResponse.json(data);
  } catch (error) {
    console.error(`[Health] Error:`, error instanceof Error ? error.message : String(error));
    return NextResponse.json(
      { error: 'Backend unavailable', details: error instanceof Error ? error.message : String(error) },
      { status: 502 }
    );
  }
}