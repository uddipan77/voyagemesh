// The trip-planning form (brief §19B). A controlled form producing a TripRequest exactly in
// the gateway's shape. Client-side constraints mirror the server's (dates, positive numbers)
// for a fast, friendly experience — but the server revalidates everything; the UI never
// assumes its own checks are authoritative.

import { useState, type FormEvent } from "react";
import type { Currency, TripRequest } from "../api/types";

const CURRENCIES: Currency[] = ["EUR", "USD", "GBP", "CHF"];
const TRANSPORT = ["any", "train", "bus", "flight", "car", "ferry"];
const ACCOMMODATION = ["any", "hostel", "budget_hotel", "hotel", "apartment", "guesthouse"];
const RANKING = ["balanced", "cheapest", "fastest", "fewest_transfers"];
const ACCESSIBILITY = [
  "step_free_access",
  "wheelchair_accessible",
  "elevator_required",
  "accessible_bathroom",
];

function today(offsetDays = 0): string {
  const d = new Date();
  d.setDate(d.getDate() + offsetDays);
  return d.toISOString().slice(0, 10);
}

const DEFAULTS = {
  origin: "Nuremberg",
  destination: "Prague",
  departure_date: today(21),
  return_date: today(24),
  travellers: 1,
  guest_nationality: "",
  budget: "350.00",
  currency: "EUR" as Currency,
  transport_preference: "any",
  accommodation_preference: "hostel",
  interests: "history, architecture, local food",
  max_transport_duration_hours: 8,
  max_transfers: 2,
  ranking_strategy: "balanced",
  accessibility: [] as string[],
};

export function TripForm({
  onSubmit,
  busy,
}: {
  onSubmit: (trip: TripRequest) => void;
  busy: boolean;
}): JSX.Element {
  const [f, setF] = useState(DEFAULTS);

  const set = <K extends keyof typeof DEFAULTS>(key: K, value: (typeof DEFAULTS)[K]) =>
    setF((prev) => ({ ...prev, [key]: value }));

  const toggleAccess = (need: string) =>
    setF((prev) => ({
      ...prev,
      accessibility: prev.accessibility.includes(need)
        ? prev.accessibility.filter((n) => n !== need)
        : [...prev.accessibility, need],
    }));

  const submit = (e: FormEvent) => {
    e.preventDefault();
    const interests = f.interests
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
    onSubmit({
      origin: f.origin.trim(),
      destination: f.destination.trim(),
      departure_date: f.departure_date,
      return_date: f.return_date,
      travellers: Number(f.travellers),
      guest_nationality: f.guest_nationality.trim().toUpperCase() || null,
      max_budget: { amount: f.budget, currency: f.currency },
      accommodation_preference: f.accommodation_preference,
      transport_preference: f.transport_preference,
      interests,
      max_transport_duration_hours: Number(f.max_transport_duration_hours),
      max_transfers: Number(f.max_transfers),
      accessibility_needs: f.accessibility,
      ranking_strategy: f.ranking_strategy,
    });
  };

  return (
    <form className="trip-form card" onSubmit={submit}>
      <h2>Plan a trip</h2>
      <div className="grid">
        <label>
          Origin
          <input value={f.origin} onChange={(e) => set("origin", e.target.value)} required />
        </label>
        <label>
          Destination
          <input
            value={f.destination}
            onChange={(e) => set("destination", e.target.value)}
            required
          />
        </label>
        <label>
          Departure
          <input
            type="date"
            value={f.departure_date}
            onChange={(e) => set("departure_date", e.target.value)}
            required
          />
        </label>
        <label>
          Return
          <input
            type="date"
            value={f.return_date}
            min={f.departure_date}
            onChange={(e) => set("return_date", e.target.value)}
            required
          />
        </label>
        <label>
          Travellers (adults)
          <input
            type="number"
            min={1}
            max={12}
            value={f.travellers}
            onChange={(e) => set("travellers", Number(e.target.value))}
          />
        </label>
        <label>
          Guest nationality (required for live hotel prices)
          <input
            value={f.guest_nationality}
            onChange={(e) => set("guest_nationality", e.target.value.toUpperCase())}
            placeholder="Two-letter code, e.g. DE or IN"
            pattern="[A-Z]{2}"
            maxLength={2}
          />
        </label>
        <label>
          Budget (total)
          <div className="money">
            <input
              inputMode="decimal"
              value={f.budget}
              onChange={(e) => set("budget", e.target.value)}
              required
            />
            <select value={f.currency} onChange={(e) => set("currency", e.target.value as Currency)}>
              {CURRENCIES.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </div>
        </label>
        <label>
          Travel mode
          <select
            value={f.transport_preference}
            onChange={(e) => set("transport_preference", e.target.value)}
          >
            {TRANSPORT.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        </label>
        <label>
          Accommodation
          <select
            value={f.accommodation_preference}
            onChange={(e) => set("accommodation_preference", e.target.value)}
          >
            {ACCOMMODATION.map((a) => (
              <option key={a} value={a}>
                {a.replace("_", " ")}
              </option>
            ))}
          </select>
        </label>
        <label>
          Max transport hours
          <input
            type="number"
            min={1}
            max={48}
            value={f.max_transport_duration_hours}
            onChange={(e) => set("max_transport_duration_hours", Number(e.target.value))}
          />
        </label>
        <label>
          Max transfers
          <input
            type="number"
            min={0}
            max={5}
            value={f.max_transfers}
            onChange={(e) => set("max_transfers", Number(e.target.value))}
          />
        </label>
        <label>
          Ranking
          <select
            value={f.ranking_strategy}
            onChange={(e) => set("ranking_strategy", e.target.value)}
          >
            {RANKING.map((r) => (
              <option key={r} value={r}>
                {r.replace("_", " ")}
              </option>
            ))}
          </select>
        </label>
        <label className="span-2">
          Interests (comma-separated)
          <input value={f.interests} onChange={(e) => set("interests", e.target.value)} />
        </label>
      </div>

      <fieldset className="accessibility">
        <legend>Accessibility</legend>
        {ACCESSIBILITY.map((need) => (
          <label key={need} className="check">
            <input
              type="checkbox"
              checked={f.accessibility.includes(need)}
              onChange={() => toggleAccess(need)}
            />
            {need.replace(/_/g, " ")}
          </label>
        ))}
      </fieldset>

      <button className="btn btn-primary btn-lg" type="submit" disabled={busy}>
        {busy ? "Planning…" : "Plan my trip"}
      </button>
    </form>
  );
}
