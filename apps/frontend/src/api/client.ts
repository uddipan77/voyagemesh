// The one place the browser talks to the gateway. Attaches the bearer token, generates an
// Idempotency-Key for plan submissions (so a double-click never plans twice), and turns a
// non-2xx response into a typed ApiError carrying the server's problem-details body.

import { config } from "../config";
import type { ErrorResponse, TripPlan, TripRequest } from "./types";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly body: ErrorResponse | null,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }

  /** A message safe and useful to show a user. */
  get userMessage(): string {
    if (this.body?.detail) return this.body.detail;
    if (this.status === 0) return "Could not reach the server.";
    return `Request failed (${this.status}).`;
  }
}

type TokenFn = () => Promise<string | null>;

async function request<T>(
  path: string,
  { method = "GET", body, token, headers = {} }: {
    method?: string;
    body?: unknown;
    token: TokenFn;
    headers?: Record<string, string>;
  },
): Promise<T> {
  const bearer = await token();
  const finalHeaders: Record<string, string> = { ...headers };
  if (body !== undefined) finalHeaders["Content-Type"] = "application/json";
  if (bearer) finalHeaders["Authorization"] = `Bearer ${bearer}`;

  let res: Response;
  try {
    res = await fetch(`${config.apiBaseUrl}${path}`, {
      method,
      headers: finalHeaders,
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
  } catch {
    throw new ApiError(0, null, "network error");
  }

  if (!res.ok) {
    let parsed: ErrorResponse | null = null;
    try {
      parsed = (await res.json()) as ErrorResponse;
    } catch {
      /* non-JSON error body — leave parsed null */
    }
    throw new ApiError(res.status, parsed, parsed?.detail ?? res.statusText);
  }
  return (await res.json()) as T;
}

/** A UUID-ish idempotency key with no dependency on crypto.randomUUID availability. */
function idempotencyKey(): string {
  if ("randomUUID" in crypto) return crypto.randomUUID();
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

export function createApi(token: TokenFn) {
  return {
    planTrip: (trip: TripRequest): Promise<TripPlan> =>
      request<TripPlan>("/api/v1/trips", {
        method: "POST",
        body: trip,
        token,
        headers: { "Idempotency-Key": idempotencyKey() },
      }),

    getTrip: (tripId: string): Promise<TripPlan> =>
      request<TripPlan>(`/api/v1/trips/${encodeURIComponent(tripId)}`, { token }),

    readiness: (): Promise<{ status: string; components: Record<string, string> }> =>
      request("/health/ready", { token }),
  };
}

export type Api = ReturnType<typeof createApi>;
