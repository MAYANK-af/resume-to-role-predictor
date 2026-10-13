import type { PredictResponse, MetadataResponse, UploadResponse } from './types';
// Read from environment variable VITE_API_URL; default to current origin in production or https://resume-to-role-predictor.onrender.com
const defaultApi = typeof window !== 'undefined' && window.location.origin.startsWith('http')
  ? (window.location.port === '3000' ? 'http://localhost:8000' : window.location.origin)
  : 'https://resume-to-role-predictor.onrender.com';
const API_BASE = (import.meta.env.VITE_API_URL || defaultApi).replace(/\/+$/, '');

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

export async function uploadResumeFile(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch(`${API_BASE}/upload-resume`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      const errorData = await res.json().catch(() => ({ detail: 'Failed to extract text from file' }));
      throw new Error(errorData.detail || `File parsing error (${res.status})`);
    }

    return await res.json();
  } catch (err: unknown) {
    // If it's a plain text or markdown file, fallback to browser FileReader if server call fails
    const ext = file.name.split('.').pop()?.toLowerCase();
    if (ext === 'txt' || ext === 'md') {
      try {
        const text = await file.text();
        if (text.trim()) {
          return {
            filename: file.name,
            text: text.trim(),
            word_count: text.trim().split(/\s+/).length,
            char_count: text.length,
            file_type: ext.toUpperCase(),
          };
        }
      } catch {
        // Fall through to re-throw original error
      }
    }
    throw err instanceof Error ? err : new Error('Failed to process file');
  }
}

