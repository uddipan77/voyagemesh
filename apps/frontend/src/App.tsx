// The application shell: header with identity, the trip form, the execution panel, and the
// result tabs. It owns the request state machine (idle → planning → result | error) and gates
// planning behind authentication.

import { useMemo, useState } from "react";
import { ApiError, createApi } from "./api/client";
import type { TripPlan, TripRequest } from "./api/types";
import { useAuth } from "./auth/AuthContext";
import { ExecutionStatus } from "./components/ExecutionStatus";
import { LoginPanel } from "./components/LoginPanel";
import { ResultTabs } from "./components/ResultTabs";
import { TripForm } from "./components/TripForm";

export function App(): JSX.Element {
  const auth = useAuth();
  const api = useMemo(() => createApi(auth.getToken), [auth.getToken]);

  const [busy, setBusy] = useState(false);
  const [plan, setPlan] = useState<TripPlan | null>(null);
  const [error, setError] = useState<ApiError | null>(null);

  const canPlan = auth.status === "authenticated" || auth.mode === "bypass";

  async function planTrip(trip: TripRequest): Promise<void> {
    setBusy(true);
    setError(null);
    setPlan(null);
    try {
      setPlan(await api.planTrip(trip));
    } catch (e) {
      setError(e instanceof ApiError ? e : new ApiError(0, null, "unexpected error"));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand">
          <span className="logo">◇</span>
          <div>
            <h1>VoyageMesh</h1>
            <p className="tagline">Observable, budget-aware multi-agent travel planning</p>
          </div>
        </div>
        <LoginPanel />
      </header>

      <main className="app-main">
        {auth.status === "loading" && <p className="muted">Signing in…</p>}
        {auth.status === "error" && (
          <p className="error-text">Authentication error: {auth.error}</p>
        )}

        {!canPlan && auth.status !== "loading" ? (
          <section className="card signin-cta">
            <h2>Sign in to plan a trip</h2>
            <p className="muted">
              VoyageMesh authenticates every request through Keycloak. Sign in to continue.
            </p>
            <button className="btn btn-primary btn-lg" onClick={auth.login}>
              Sign in with Keycloak
            </button>
          </section>
        ) : (
          <>
            <TripForm onSubmit={planTrip} busy={busy} />
            <ExecutionStatus busy={busy} plan={plan} error={error} />
            {plan && <ResultTabs plan={plan} />}
          </>
        )}
      </main>

      <footer className="app-footer">
        <span>VoyageMesh · portfolio project</span>
        <span className="muted">
          Data labels: LIVE / CACHED / MOCKED / FIXTURE / COMPUTED / UNAVAILABLE
        </span>
      </footer>
    </div>
  );
}
