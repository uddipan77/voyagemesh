// Runtime configuration, read once from Vite's import.meta.env. Every value has a sensible
// local-development default so the app runs with no .env at all.

export type AuthMode = "oidc" | "bypass";

const env = import.meta.env;

export const config = {
  apiBaseUrl: (env.VITE_API_BASE_URL as string | undefined) ?? "",
  authMode: ((env.VITE_AUTH_MODE as string | undefined) ?? "oidc") as AuthMode,
  oidc: {
    authority: (env.VITE_OIDC_AUTHORITY as string | undefined) ??
      "http://localhost:8080/realms/voyagemesh",
    clientId: (env.VITE_OIDC_CLIENT_ID as string | undefined) ?? "voyagemesh-frontend",
    redirectUri: (env.VITE_OIDC_REDIRECT_URI as string | undefined) ??
      `${window.location.origin}/`,
    scope: "openid profile email roles",
  },
} as const;
