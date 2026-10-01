// The single source of truth for "who is signed in and how do we get a token". Wraps the two
// modes — real Keycloak (oidc) and the local dev bypass — behind one interface so the rest of
// the app never branches on the mode.

import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { config } from "../config";
import * as oidc from "./oidc";
import type { UserClaims } from "./oidc";

type Status = "loading" | "authenticated" | "anonymous" | "error";

interface AuthState {
  status: Status;
  mode: "oidc" | "bypass";
  user: UserClaims | null;
  error: string | null;
  login: () => void;
  logout: () => void;
  /** A bearer token for API calls, or null in bypass mode (the gateway's dev bypass applies). */
  getToken: () => Promise<string | null>;
}

const BYPASS_USER: UserClaims = {
  subject: "dev-bypass-user",
  username: "dev (bypass)",
  roles: ["traveller", "evaluator", "admin"],
};

const AuthContext = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: ReactNode }): JSX.Element {
  const bypass = config.authMode === "bypass";
  const [status, setStatus] = useState<Status>(bypass ? "authenticated" : "loading");
  const [user, setUser] = useState<UserClaims | null>(bypass ? BYPASS_USER : null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (bypass) return;
    let cancelled = false;
    (async () => {
      try {
        await oidc.completeLoginIfCallback();
        const token = await oidc.getAccessToken();
        if (cancelled) return;
        if (token) {
          setUser(oidc.currentClaims());
          setStatus("authenticated");
        } else {
          setStatus("anonymous");
        }
      } catch (e) {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : "authentication failed");
        setStatus("error");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [bypass]);

  const value = useMemo<AuthState>(
    () => ({
      status,
      mode: bypass ? "bypass" : "oidc",
      user,
      error,
      login: () => {
        void oidc.login();
      },
      logout: () => {
        if (bypass) return;
        void oidc.logout();
      },
      getToken: async () => (bypass ? null : oidc.getAccessToken()),
    }),
    [status, user, error, bypass],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within an AuthProvider");
  return ctx;
}
