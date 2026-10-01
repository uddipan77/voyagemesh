// TypeScript mirrors of the gateway contracts. The top-level shapes (TripRequest, TripPlan,
// ErrorResponse, DataSourceInfo) are modelled precisely because the UI branches on them; the
// deep section payloads (transport/accommodation/itinerary/budget) are typed loosely and
// rendered defensively, so a schema addition on the server never breaks the client.

export type Currency = "EUR" | "USD" | "GBP" | "CHF" | "JPY";

export type PlanStatus = "complete" | "partial" | "no_viable_plan" | "failed";

export type DataOrigin =
  | "live"
  | "cached"
  | "mocked"
  | "fixture"
  | "computed"
  | "unavailable";

export interface Money {
  amount: number | string;
  currency: Currency;
}

export interface TripRequest {
  origin: string;
  destination: string;
  departure_date: string; // YYYY-MM-DD
  return_date: string;
  travellers: number;
  guest_nationality?: string | null;
  max_budget: Money;
  accommodation_preference: string;
  transport_preference: string;
  interests: string[];
  max_transport_duration_hours: number;
  max_transfers: number;
  accessibility_needs: string[];
  ranking_strategy: string;
  notes?: string | null;
}

export interface DataSourceInfo {
  component: string;
  source_name: string;
  origin: DataOrigin;
  retrieved_at?: string | null;
  note?: string | null;
  label: string;
}

export interface TripPlan {
  request_id: string;
  trip_id: string;
  status: PlanStatus;
  generated_at: string;
  transport?: SectionPayload | null;
  accommodation?: SectionPayload | null;
  itinerary?: SectionPayload | null;
  budget?: SectionPayload | null;
  trade_off_explanation: string;
  reasoning_summary: string;
  data_sources: DataSourceInfo[];
  confidence: string;
  completeness_percent: number;
  degraded_services: string[];
  warnings: string[];
  unavailable_sections: string[];
  replans: number;
  cache_status: string;
  validation_errors: string[];
  is_usable?: boolean;
  within_budget?: boolean | null;
}

// Deliberately permissive — see the module note above.
export type SectionPayload = Record<string, unknown>;

export interface ErrorResponse {
  title: string;
  status: number;
  code: string;
  detail: string;
  request_id?: string | null;
  trace_id?: string | null;
  problems?: { field: string; message: string; provided?: string | null }[];
  retry_after_seconds?: number | null;
}
