// A small OIDC Authorization-Code-with-PKCE client for Keycloak, with zero runtime
// dependencies. It discovers the realm's endpoints, drives the redirect login, exchanges the
// code for tokens, refreshes them, and decodes claims for the UI.
//
// Design notes:
//  * The code_verifier and state live in sessionStorage only for the brief round-trip to
//    Keycloak and back; tokens live in memory-backed sessionStorage so a refresh survives a
//    page reload but nothing is written to long-lived storage.
//  * We never handle a client secret — this is a public client.

import { config } from "../config";
import { challengeFromVerifier, randomString } from "./pkce";

export interface UserClaims {
  subject: string;
  username: string;
  email?: string;
  roles: string[];
}

interface TokenSet {
  accessToken: string;
  refreshToken?: string;
  idToken?: string;
  expiresAt: number; // epoch ms
}

interface Endpoints {
  authorization_endpoint: string;
  token_endpoint: string;
  end_session_endpoint?: string;
}

const STORAGE_TOKENS = "vm.tokens";
const STORAGE_VERIFIER = "vm.pkce.verifier";
const STORAGE_STATE = "vm.pkce.state";

let endpointsCache: Endpoints | null = null;

async function discover(): Promise<Endpoints> {
  if (endpointsCache) return endpointsCache;
  const res = await fetch(`${config.oidc.authority}/.well-known/openid-configuration`);
  if (!res.ok) throw new Error(`OIDC discovery failed (${res.status})`);
  endpointsCache = (await res.json()) as Endpoints;
  return endpointsCache;
}

function loadTokens(): TokenSet | null {
  const raw = sessionStorage.getItem(STORAGE_TOKENS);
  return raw ? (JSON.parse(raw) as TokenSet) : null;
}

function saveTokens(tokens: TokenSet): void {
  sessionStorage.setItem(STORAGE_TOKENS, JSON.stringify(tokens));
}

export function clearSession(): void {
  sessionStorage.removeItem(STORAGE_TOKENS);
  sessionStorage.removeItem(STORAGE_VERIFIER);
  sessionStorage.removeItem(STORAGE_STATE);
}

export function decodeClaims(accessToken: string): UserClaims {
  const payloadPart = accessToken.split(".")[1] ?? "";
  const json = atob(payloadPart.replace(/-/g, "+").replace(/_/g, "/"));
  const claims = JSON.parse(json) as Record<string, unknown>;
  const realmRoles = (claims.realm_access as { roles?: string[] } | undefined)?.roles ?? [];
  const resource = claims.resource_access as Record<string, { roles?: string[] }> | undefined;
  const clientRoles = resource?.[config.oidc.clientId]?.roles ?? [];
  return {
    subject: String(claims.sub ?? ""),
    username: String(claims.preferred_username ?? claims.sub ?? "unknown"),
    email: claims.email ? String(claims.email) : undefined,
    roles: Array.from(new Set([...realmRoles, ...clientRoles])),
  };
}

/** Begin login: redirect the browser to Keycloak. */
export async function login(): Promise<void> {
  const endpoints = await discover();
  const verifier = randomString();
  const state = randomString(16);
  sessionStorage.setItem(STORAGE_VERIFIER, verifier);
  sessionStorage.setItem(STORAGE_STATE, state);
  const params = new URLSearchParams({
    response_type: "code",
    client_id: config.oidc.clientId,
    redirect_uri: config.oidc.redirectUri,
    scope: config.oidc.scope,
    state,
    code_challenge: await challengeFromVerifier(verifier),
    code_challenge_method: "S256",
  });
  window.location.assign(`${endpoints.authorization_endpoint}?${params.toString()}`);
}

/** If the current URL is a redirect callback, complete the code exchange. Returns true if a
 *  login was completed (and strips the query params from the URL). */
export async function completeLoginIfCallback(): Promise<boolean> {
  const url = new URL(window.location.href);
  const code = url.searchParams.get("code");
  const returnedState = url.searchParams.get("state");
  if (!code) return false;

  const expectedState = sessionStorage.getItem(STORAGE_STATE);
  const verifier = sessionStorage.getItem(STORAGE_VERIFIER);
  if (!verifier || !expectedState || returnedState !== expectedState) {
    clearSession();
    throw new Error("login state mismatch — please try again");
  }

  const endpoints = await discover();
  const body = new URLSearchParams({
    grant_type: "authorization_code",
    client_id: config.oidc.clientId,
    code,
    redirect_uri: config.oidc.redirectUri,
    code_verifier: verifier,
  });
  const res = await fetch(endpoints.token_endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body,
  });
  if (!res.ok) throw new Error(`token exchange failed (${res.status})`);
  saveTokens(toTokenSet(await res.json()));

  sessionStorage.removeItem(STORAGE_VERIFIER);
  sessionStorage.removeItem(STORAGE_STATE);
  // Remove ?code&state from the address bar so a reload doesn't re-trigger the exchange.
  window.history.replaceState({}, document.title, url.pathname);
  return true;
}

/** Return a valid access token, refreshing it if it is within 30s of expiry. Null if signed
 *  out or the refresh failed. */
export async function getAccessToken(): Promise<string | null> {
  const tokens = loadTokens();
  if (!tokens) return null;
  if (Date.now() < tokens.expiresAt - 30_000) return tokens.accessToken;
  if (!tokens.refreshToken) return null;

  try {
    const endpoints = await discover();
    const res = await fetch(endpoints.token_endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({
        grant_type: "refresh_token",
        client_id: config.oidc.clientId,
        refresh_token: tokens.refreshToken,
      }),
    });
    if (!res.ok) {
      clearSession();
      return null;
    }
    const refreshed = toTokenSet(await res.json());
    saveTokens(refreshed);
    return refreshed.accessToken;
  } catch {
    return null;
  }
}

export function currentClaims(): UserClaims | null {
  const tokens = loadTokens();
  return tokens ? decodeClaims(tokens.accessToken) : null;
}

export async function logout(): Promise<void> {
  const tokens = loadTokens();
  clearSession();
  const endpoints = await discover();
  if (endpoints.end_session_endpoint) {
    const params = new URLSearchParams({ post_logout_redirect_uri: config.oidc.redirectUri });
    if (tokens?.idToken) params.set("id_token_hint", tokens.idToken);
    window.location.assign(`${endpoints.end_session_endpoint}?${params.toString()}`);
  } else {
    window.location.reload();
  }
}

interface RawTokenResponse {
  access_token: string;
  refresh_token?: string;
  id_token?: string;
  expires_in?: number;
}

function toTokenSet(raw: RawTokenResponse): TokenSet {
  return {
    accessToken: raw.access_token,
    refreshToken: raw.refresh_token,
    idToken: raw.id_token,
    expiresAt: Date.now() + (raw.expires_in ?? 300) * 1000,
  };
}
