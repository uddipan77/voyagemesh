// The execution panel (brief §19C). Planning is synchronous — the gateway returns one complete
// plan — so this panel is honest about what it knows: while the request is in flight it shows
// the orchestrator and agents as "running"; once the plan returns it derives each agent's
// outcome from the sections that actually arrived, plus cache status, replans, and warnings.
// It never invents a live progress stream it does not have.

import type { TripPlan } from "../api/types";
import { ApiError } from "../api/client";

type AgentState = "running" | "completed" | "failed" | "idle";

const AGENTS: { key: "transport" | "accommodation" | "itinerary"; label: string }[] = [
  { key: "transport", label: "Transport Agent" },
  { key: "accommodation", label: "Stay Agent" },
  { key: "itinerary", label: "Itinerary Agent" },
];

function agentState(plan: TripPlan | null, busy: boolean, key: string): AgentState {
  if (busy) return "running";
  if (!plan) return "idle";
  const section = plan[key as keyof TripPlan];
  return section ? "completed" : "failed";
}

function Dot({ state }: { state: AgentState }): JSX.Element {
  return <span className={`dot dot-${state}`} aria-label={state} />;
}

export function ExecutionStatus({
  busy,
  plan,
  error,
}: {
  busy: boolean;
  plan: TripPlan | null;
  error: ApiError | null;
}): JSX.Element | null {
  if (!busy && !plan && !error) return null;

  return (
    <section className="card execution">
      <h3>Execution</h3>
      <ul className="steps">
        <li>
          <Dot state={busy ? "running" : plan || error ? "completed" : "idle"} />
          Orchestrator {busy ? "running" : error ? "stopped" : "completed"}
        </li>
        {AGENTS.map(({ key, label }) => {
          const state = error ? "failed" : agentState(plan, busy, key);
          return (
            <li key={key}>
              <Dot state={state} />
              {label} <span className="muted">{state}</span>
            </li>
          );
        })}
      </ul>

      {plan && (
        <div className="exec-meta">
          <span className="pill">cache: {plan.cache_status}</span>
          <span className="pill">replans: {plan.replans}</span>
          <span className="pill">complete: {plan.completeness_percent}%</span>
          {plan.degraded_services.map((s) => (
            <span key={s} className="pill pill-warn">
              degraded: {s}
            </span>
          ))}
        </div>
      )}

      {plan?.warnings?.length ? (
        <ul className="warnings">
          {plan.warnings.map((w, i) => (
            <li key={i}>⚠ {w}</li>
          ))}
        </ul>
      ) : null}

      {error && <p className="error-text">✕ {error.userMessage}</p>}
    </section>
  );
}
