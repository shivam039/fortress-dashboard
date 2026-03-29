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

    const response = await fetch(url);
    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json({ error: 'Backend unavailable' }, { status: 500 });
  }
}