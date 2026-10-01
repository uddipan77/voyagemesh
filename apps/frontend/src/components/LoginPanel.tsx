// The header's identity area: sign-in button when anonymous, current user + roles + sign-out
// when authenticated. In bypass mode it shows a clear banner that requests are unauthenticated.

import { useAuth } from "../auth/AuthContext";

export function LoginPanel(): JSX.Element {
  const auth = useAuth();

  if (auth.mode === "bypass") {
    return (
      <div className="login-panel">
        <span className="pill pill-warn" title="Keycloak is bypassed; requests are unauthenticated">
          DEV BYPASS
        </span>
        <span className="user-name">{auth.user?.username}</span>
      </div>
    );
  }

  if (auth.status === "authenticated" && auth.user) {
    return (
      <div className="login-panel">
        <div className="user-block">
          <span className="user-name">{auth.user.username}</span>
          <span className="user-roles">
            {auth.user.roles.length ? auth.user.roles.join(" · ") : "no roles"}
          </span>
        </div>
        <button className="btn btn-ghost" onClick={auth.logout}>
          Sign out
        </button>
      </div>
    );
  }

  return (
    <div className="login-panel">
      <button className="btn btn-primary" onClick={auth.login}>
        Sign in with Keycloak
      </button>
    </div>
  );
}
