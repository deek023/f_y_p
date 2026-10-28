/**
 * KalaKriti — Neural Style Transfer API Client
 * Conforms strictly to the backend API Contract:
 *   - GET /health
 *   - POST /stylize
 *
 * Supports dynamic backend URL configuration (e.g. Google Colab ngrok tunnel or local server).
 */

export interface HealthResponse {
  status: 'healthy' | 'model_not_ready' | 'degraded' | string;
  engine: string;
  device: string;
  modelReady: boolean;
  loadedStyles: string[];
  message?: string;
  version: string;
}

export interface StylizeRequest {
  contentImageBase64: string;
  styleId: string;
  styleStrength: number;
  resolution: number;
}

export interface StylizeResponse {
  stylizedImageBase64: string;
  inferenceTimeMs: number;
  styleId: string;
  resolution: [number, number];
}

const STORAGE_KEY_BACKEND_URL = 'kalakriti_backend_url';
export const DEFAULT_BACKEND_URL = 'https://overtake-browse-ninja.ngrok-free.dev';

export function getBackendUrl(): string {
  if (typeof window === 'undefined') return DEFAULT_BACKEND_URL;
  const saved = localStorage.getItem(STORAGE_KEY_BACKEND_URL);
  return saved ? saved.trim().replace(/\/+$/, '') : DEFAULT_BACKEND_URL;
}

export function setBackendUrl(url: string): void {
  const cleanUrl = url.trim().replace(/\/+$/, '');
  localStorage.setItem(STORAGE_KEY_BACKEND_URL, cleanUrl);
}

export async function checkHealth(overrideUrl?: string): Promise<HealthResponse> {
  const baseUrl = overrideUrl ? overrideUrl.trim().replace(/\/+$/, '') : getBackendUrl();
  const endpoint = `${baseUrl}/health`;

  try {
    const res = await fetch(endpoint, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
        'ngrok-skip-browser-warning': 'true',
      },
      // Short timeout to detect disconnected status quickly
      signal: AbortSignal.timeout(6000),
    });

    if (!res.ok) {
      throw new Error(`Server returned HTTP ${res.status}: ${res.statusText}`);
    }

    const data: HealthResponse = await res.json();
    return data;
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    console.error(`[KalaKriti API] Health check failed for ${endpoint}:`, err);
    throw new Error(
      `Cannot connect to KalaKriti backend at ${baseUrl}. Ensure your FastAPI server (or Colab ngrok tunnel) is running and CORS is enabled. (${message})`
    );
  }
}

export async function stylizeImage(
  req: StylizeRequest,
  overrideUrl?: string
): Promise<StylizeResponse> {
  const baseUrl = overrideUrl ? overrideUrl.trim().replace(/\/+$/, '') : getBackendUrl();
  const endpoint = `${baseUrl}/stylize`;

  try {
    const res = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'ngrok-skip-browser-warning': 'true',
      },
      body: JSON.stringify(req),
    });

    if (!res.ok) {
      let errorDetail = `HTTP ${res.status}`;
      try {
        const errorJson = await res.json();
        if (errorJson.detail) {
          errorDetail = errorJson.detail;
        }
      } catch {
        errorDetail = await res.text();
      }
      throw new Error(errorDetail);
    }

    const data: StylizeResponse = await res.json();
    return data;
  } catch (err: unknown) {
    console.error(`[KalaKriti API] Stylize execution failed at ${endpoint}:`, err);
    if (err instanceof Error) {
      throw err;
    }
    throw new Error(String(err));
  }
}
