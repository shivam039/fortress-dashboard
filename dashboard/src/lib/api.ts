/**
 * Optimized fetch utility for proxying requests to the backend with timeout and retries.
 */

export const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || process.env.BACKEND_URL ||
  (process.env.NODE_ENV === 'production'
    ? 'https://shivam039-dev-fortress-engine.hf.space'
    : 'http://localhost:8000');

export async function fetchWithTimeout(url: string, options: any = {}, timeout = 15000) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeout);

  // Hugging Face Spaces often require a User-Agent and specific Accept headers
  const headers = {
    'User-Agent': 'FortressDashboard-Vercel/1.0',
    'Accept': 'application/json',
    ...(options.headers || {}),
  };

  try {
    const response = await fetch(url, {
      ...options,
      headers,
      signal: controller.signal,
    });
    clearTimeout(id);
    return response;
  } catch (error) {
    clearTimeout(id);
    throw error;
  }
}

export async function fetchWithRetry(url: string, options: any = {}, retries = 2) {
  let lastError: any;

  for (let i = 0; i <= retries; i++) {
    try {
      if (i > 0) {
        console.log(`[API Proxy] Retry ${i}/${retries} for ${url}`);
        // Exponential backoff
        await new Promise(r => setTimeout(r, 500 * i));
      }

      const response = await fetchWithTimeout(url, options);
      
      // If we got a response, even if it's 404 or 500, we return it.
      if (response.ok || i === retries) {
        return response;
      }
    } catch (error: any) {
      lastError = error;
      console.error(`[API Proxy] Attempt ${i + 1} failed for ${url}:`, error.message || error);
      
      if (i === retries) throw error;
    }
  }
  
  throw lastError;
}
