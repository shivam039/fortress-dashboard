import { NextResponse } from 'next/server';

export async function GET() {
  try {
    const testUrl = 'https://shivam039-dev-fortress-engine.hf.space/api/health';
    console.log(`[Test] Starting test at ${new Date().toISOString()}`);
    console.log(`[Test] Testing URL: ${testUrl}`);
    
    const abortController = new AbortController();
    const timeoutId = setTimeout(() => abortController.abort(), 8000);
    
    const startTime = Date.now();
    const response = await fetch(testUrl, {
      signal: abortController.signal,
      headers: {
        'User-Agent': 'Vercel-Test/1.0',
      },
    });
    const endTime = Date.now();
    clearTimeout(timeoutId);
    
    console.log(`[Test] Response received in ${endTime - startTime}ms, status: ${response.status}`);
    
    const contentType = response.headers.get('content-type');
    console.log(`[Test] Content-Type: ${contentType}`);
    
    let body;
    if (contentType?.includes('application/json')) {
      body = await response.json();
    } else {
      body = await response.text();
    }
    
    return NextResponse.json({
      success: true,
      message: 'Backend is reachable',
      backend_response: body,
      status: response.status,
      response_time_ms: endTime - startTime,
      environment: process.env.NODE_ENV,
    });
  } catch (error) {
    console.error(`[Test] Error:`, error);
    return NextResponse.json({
      success: false,
      message: 'Backend unreachable',
      error: error instanceof Error ? error.message : String(error),
      error_type: error instanceof Error ? error.constructor.name : typeof error,
      environment: process.env.NODE_ENV,
      timestamp: new Date().toISOString(),
    }, { status: 503 });
  }
}
