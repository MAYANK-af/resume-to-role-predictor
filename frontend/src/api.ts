import type { PredictResponse, MetadataResponse } from './types';
// Read from environment variable VITE_API_URL with http://localhost:8000 as default
const API_BASE = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/+$/, '');

export async function checkApiHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(3000) });
    return res.ok;
  } catch {
    return false;
  }
}

export async function fetchMetadata(): Promise<MetadataResponse | null> {
  try {
    const res = await fetch(`${API_BASE}/metadata`);
    if (!res.ok) throw new Error('Failed to load metadata');
    return await res.json();
  } catch (err) {
    console.error('Metadata fetch error:', err);
    return null;
  }
}

export async function fetchSamples(): Promise<Record<string, string>> {
  try {
    const res = await fetch(`${API_BASE}/samples`);
    if (!res.ok) throw new Error('Failed to load samples');
    return await res.json();
  } catch (err) {
    console.error('Samples fetch error:', err);
    return {};
  }
}

export async function sendPrediction(
  resumeText: string,
  targetRole?: string
): Promise<PredictResponse> {
  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      resume_text: resumeText,
      target_role: targetRole || null,
    }),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: 'Analysis failed' }));
    throw new Error(errorData.detail || `Server error (${res.status})`);
  }

  return await res.json();
}
