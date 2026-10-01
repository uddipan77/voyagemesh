// The result view (brief §19D): Overview, Transport, Accommodation, Itinerary, Budget, Sources,
// Agent activity, and Observability links. Each tab renders defensively from the plan and shows
// honest labels. Private agent reasoning is never shown — only concise summaries the plan
// carries deliberately.

import { useState } from "react";
import type { TripPlan } from "../api/types";
import { DataLabel } from "./DataLabel";
import { ObjectCard, RawJson, asArray, asRecord } from "./render";

type TabKey =
  | "overview"
  | "transport"
  | "accommodation"
  | "itinerary"
  | "budget"
  | "sources"
  | "activity"
  | "observability";

const TABS: { key: TabKey; label: string }[] = [
  { key: "overview", label: "Overview" },
  { key: "transport", label: "Transport" },
  { key: "accommodation", label: "Accommodation" },
  { key: "itinerary", label: "Itinerary" },
  { key: "budget", label: "Budget" },
  { key: "sources", label: "Sources" },
  { key: "activity", label: "Agent activity" },
  { key: "observability", label: "Observability" },
];

export function ResultTabs({ plan }: { plan: TripPlan }): JSX.Element {
  const [tab, setTab] = useState<TabKey>("overview");
  return (
    <section className="card results">
      <div className="tabbar" role="tablist">
        {TABS.map((t) => (
          <button
            key={t.key}
            role="tab"
            aria-selected={tab === t.key}
            className={`tab ${tab === t.key ? "tab-active" : ""}`}
            onClick={() => setTab(t.key)}
          >
            {t.label}
          </button>
        ))}
      </div>
      <div className="tab-panel" role="tabpanel">
        {tab === "overview" && <Overview plan={plan} />}
        {tab === "transport" && <OfferSection payload={plan.transport} noun="transport option" />}
        {tab === "accommodation" && (
          <OfferSection payload={plan.accommodation} noun="stay" />
        )}
        {tab === "itinerary" && <Itinerary plan={plan} />}
        {tab === "budget" && <Budget plan={plan} />}
        {tab === "sources" && <Sources plan={plan} />}
        {tab === "activity" && <Activity plan={plan} />}
        {tab === "observability" && <Observability plan={plan} />}
      </div>
    </section>
  );
}

function StatusPills({ plan }: { plan: TripPlan }): JSX.Element {
  const incompleteBudget = asRecord(plan.budget)?.status === "incomplete";
  return (
    <div className="pill-row">
      <DataLabel kind={plan.status === "partial" ? "partial" : plan.status} />
      <span className="pill">confidence: {plan.confidence}</span>
      {plan.within_budget != null && (
        <span className={`pill ${plan.within_budget ? "pill-ok" : "pill-warn"}`}>
          {incompleteBudget ? "budget incomplete" : plan.within_budget ? "within budget" : "over budget"}
        </span>
      )}
      <span className="pill">{plan.completeness_percent}% complete</span>
    </div>
  );
}

function Overview({ plan }: { plan: TripPlan }): JSX.Element {
  return (
    <div>
      <StatusPills plan={plan} />
      {plan.trade_off_explanation && (
        <>
          <h4>Trade-offs</h4>
          <p>{plan.trade_off_explanation}</p>
        </>
      )}
      {plan.reasoning_summary && (
        <>
          <h4>Summary</h4>
          <p>{plan.reasoning_summary}</p>
        </>
      )}
      {plan.unavailable_sections.length > 0 && (
        <p className="muted">
          Unavailable: {plan.unavailable_sections.join(", ")}{" "}
          <DataLabel kind="unavailable" />
        </p>
      )}
      {plan.warnings.length > 0 && (
        <ul className="warnings">
          {plan.warnings.map((warning, i) => <li key={i}>{warning}</li>)}
        </ul>
      )}
      {plan.validation_errors.length > 0 && (
        <ul className="warnings">
          {plan.validation_errors.map((e, i) => (
            <li key={i}>✕ {e}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

/** Transport and accommodation share a shape: a recommended offer plus alternatives. */
function OfferSection({
  payload,
  noun,
}: {
  payload: TripPlan["transport"];
  noun: string;
}): JSX.Element {
  const record = asRecord(payload);
  if (!record) return <Empty noun={noun} />;
  const recommended = asRecord(record.recommended);
  const alternatives = asArray(record.alternatives);
  if (!recommended) return <Empty noun={noun} />;

  return (
    <div>
      <OfferCard title="Recommended" data={recommended} />
      {alternatives.length > 0 && (
        <>
          <h4>Alternatives ({alternatives.length})</h4>
          <div className="card-grid">
            {alternatives.map((alt, i) => {
              const r = asRecord(alt);
              return r ? <OfferCard key={i} title={`Option ${i + 2}`} data={r} /> : null;
            })}
          </div>
        </>
      )}
      <RawJson data={payload} />
    </div>
  );
}

function OfferCard({ title, data }: { title: string; data: Record<string, unknown> }): JSX.Element {
  const provenance = asRecord(data.provenance);
  return (
    <div>
      {provenance && (
        <p><DataLabel kind={String(provenance.origin)} /> {String(provenance.source_name)}</p>
      )}
      <ObjectCard title={title} data={data} />
      {["legs", "return_legs"].map((key) => {
        const legs = asArray(data[key]);
        return legs.length > 0 ? (
          <div key={key}>
            <h4>{key === "legs" ? "Outbound journey" : "Return journey (included in total)"}</h4>
            {legs.map((leg, i) => {
              const row = asRecord(leg);
              return row ? <ObjectCard key={i} title={`Operated by ${String(row.carrier)}`} data={row} /> : null;
            })}
          </div>
        ) : null;
      })}
    </div>
  );
}

function Itinerary({ plan }: { plan: TripPlan }): JSX.Element {
  const record = asRecord(plan.itinerary);
  if (!record) return <Empty noun="itinerary" />;
  const itinerary = asRecord(record.itinerary);
  const days = asArray(itinerary?.days);
  const weather = asRecord(record.weather);

  return (
    <div>
      {weather && (
        <div className="weather">
          <h4>
            Weather <DataLabel kind={String(record.weather_origin ?? "unavailable")} />
          </h4>
          <ObjectCard title="Forecast" data={weather} />
        </div>
      )}
      {days.length > 0 ? (
        <ol className="day-list">
          {days.map((day, i) => {
            const d = asRecord(day);
            return (
              <li key={i}>
                <ObjectCard title={`Day ${i + 1}`} data={d ?? {}} />
              </li>
            );
          })}
        </ol>
      ) : (
        <Empty noun="itinerary" />
      )}
      <RawJson data={plan.itinerary} />
    </div>
  );
}

function Budget({ plan }: { plan: TripPlan }): JSX.Element {
  const record = asRecord(plan.budget);
  if (!record) return <Empty noun="budget" />;
  const lines = asArray(record.lines ?? record.line_items);
  return (
    <div>
      <StatusPills plan={plan} />
      <ObjectCard title="Totals" data={record} />
      {lines.length > 0 && (
        <>
          <h4>Line items</h4>
          <div className="card-grid">
            {lines.map((line, i) => {
              const l = asRecord(line);
              return l ? <ObjectCard key={i} title={`Item ${i + 1}`} data={l} /> : null;
            })}
          </div>
        </>
      )}
      <RawJson data={plan.budget} />
    </div>
  );
}

function Sources({ plan }: { plan: TripPlan }): JSX.Element {
  if (!plan.data_sources.length) return <p className="muted">No data sources recorded.</p>;
  return (
    <table className="sources-table">
      <thead>
        <tr>
          <th>Component</th>
          <th>Source</th>
          <th>Origin</th>
          <th>Retrieved</th>
        </tr>
      </thead>
      <tbody>
        {plan.data_sources.map((s, i) => (
          <tr key={i}>
            <td>{s.component}</td>
            <td>
              {s.source_name}
              {s.note && <div className="muted">{s.note}</div>}
            </td>
            <td>
              <DataLabel kind={s.origin} />
            </td>
            <td className="muted">
              {s.retrieved_at ? new Date(s.retrieved_at).toLocaleString() : "—"}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function Activity({ plan }: { plan: TripPlan }): JSX.Element {
  // Honest, non-private summary derived from the plan: which agent produced which section, and
  // which dependencies degraded. Private chain-of-thought is never surfaced (brief §19E).
  const rows = [
    { agent: "Transport Agent", section: "transport", produced: !!plan.transport },
    { agent: "Stay Agent", section: "accommodation", produced: !!plan.accommodation },
    { agent: "Itinerary Agent", section: "itinerary", produced: !!plan.itinerary },
  ];
  return (
    <div>
      <table className="sources-table">
        <thead>
          <tr>
            <th>Agent</th>
            <th>Section</th>
            <th>Outcome</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.section}>
              <td>{r.agent}</td>
              <td>{r.section}</td>
              <td>
                {r.produced ? (
                  <span className="pill pill-ok">completed</span>
                ) : (
                  <span className="pill pill-warn">no result</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="muted">
        Replans: {plan.replans} · Degraded: {plan.degraded_services.join(", ") || "none"}
      </p>
    </div>
  );
}

function Observability({ plan }: { plan: TripPlan }): JSX.Element {
  // Deep links to the local observability stack (provisioned in Phase 14 / 16). The trace and
  // request IDs correlate this plan with its server-side spans and logs.
  const links = [
    { label: "Grafana dashboards", href: "http://localhost:3001" },
    { label: "Prometheus", href: "http://localhost:9090" },
    { label: "Traces (Jaeger)", href: "http://localhost:16686" },
  ];
  return (
    <div>
      <ObjectCard
        title="Correlation"
        data={{ request_id: plan.request_id, trip_id: plan.trip_id, generated_at: plan.generated_at }}
      />
      <ul className="link-list">
        {links.map((l) => (
          <li key={l.href}>
            <a href={l.href} target="_blank" rel="noreferrer">
              {l.label} ↗
            </a>
          </li>
        ))}
      </ul>
      <p className="muted">Quote the request ID above when reporting an issue.</p>
    </div>
  );
}

function Empty({ noun }: { noun: string }): JSX.Element {
  return (
    <p className="muted">
      No {noun} was produced for this plan. <DataLabel kind="unavailable" />
    </p>
  );
}
