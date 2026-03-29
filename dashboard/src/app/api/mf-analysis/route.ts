import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const limit = searchParams.get('limit');

    const url = limit
      ? `http://localhost:8000/api/mf-analysis?limit=${limit}`
      : 'http://localhost:8000/api/mf-analysis';

    const response = await fetch(url);
    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json({ error: 'Backend unavailable' }, { status: 500 });
  }
}