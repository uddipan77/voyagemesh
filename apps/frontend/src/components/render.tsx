// Defensive renderers for the deep section payloads. The server owns the exact schema of a
// transport offer or an itinerary day; the client renders whatever scalar fields are present
// and recurses shallowly into nested objects, so a new field appears automatically and a
// missing one simply doesn't render. A collapsible raw-JSON view is always available as the
// ground truth.

const HIDDEN_KEYS = new Set(["has_result"]);

function titleCase(key: string): string {
  return key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function isMoney(v: unknown): v is { amount: unknown; currency: string } {
  return typeof v === "object" && v !== null && "amount" in v && "currency" in v;
}

export function formatValue(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (isMoney(value)) return `${value.amount} ${value.currency}`;
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (typeof value === "number" || typeof value === "string") return String(value);
  return "";
}

/** Render an object's scalar fields as a definition list; skip nested structures (those are
 *  handled by the caller or shown in the raw view). */
export function Fields({ data }: { data: Record<string, unknown> }): JSX.Element {
  const rows = Object.entries(data).filter(
    ([key, v]) =>
      !HIDDEN_KEYS.has(key) &&
      (v === null ||
        typeof v === "string" ||
        typeof v === "number" ||
        typeof v === "boolean" ||
        isMoney(v)),
  );
  if (!rows.length) return <p className="muted">No scalar fields.</p>;
  return (
    <dl className="fields">
      {rows.map(([key, v]) => (
        <div key={key} className="field-row">
          <dt>{titleCase(key)}</dt>
          <dd>{formatValue(v)}</dd>
        </div>
      ))}
    </dl>
  );
}

/** A titled card wrapping a single object's scalar fields. */
export function ObjectCard({
  title,
  data,
}: {
  title: string;
  data: Record<string, unknown>;
}): JSX.Element {
  return (
    <div className="object-card">
      <h4>{title}</h4>
      <Fields data={data} />
    </div>
  );
}

export function RawJson({ data }: { data: unknown }): JSX.Element {
  return (
    <details className="raw-json">
      <summary>Raw data</summary>
      <pre>{JSON.stringify(data, null, 2)}</pre>
    </details>
  );
}

export function asRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null ? (value as Record<string, unknown>) : null;
}

export function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}
