// The honesty labels (brief §19E). Every piece of data the plan shows is tagged with where it
// came from, so a viewer can never mistake a mocked price for a live one. The label text and
// colour are derived from a fixed vocabulary — an unknown origin renders neutrally rather than
// silently looking authoritative.

import type { DataOrigin } from "../api/types";

type LabelKind = DataOrigin | "partial" | string;

const STYLES: Record<string, { bg: string; fg: string; title: string }> = {
  live: { bg: "#0f5132", fg: "#d1e7dd", title: "Fetched live from an external provider" },
  cached: { bg: "#664d03", fg: "#fff3cd", title: "Served from cache (previously fetched)" },
  mocked: { bg: "#842029", fg: "#f8d7da", title: "Synthetic stand-in data — not real" },
  fixture: { bg: "#41464b", fg: "#e2e3e5", title: "Fixed test fixture data" },
  computed: { bg: "#084298", fg: "#cfe2ff", title: "Deterministically computed by the system" },
  unavailable: { bg: "#41464b", fg: "#e2e3e5", title: "This data could not be obtained" },
  partial: { bg: "#664d03", fg: "#fff3cd", title: "Some sections are incomplete" },
};

export function DataLabel({ kind }: { kind: LabelKind }): JSX.Element {
  const key = String(kind).toLowerCase();
  const style = STYLES[key] ?? { bg: "#41464b", fg: "#e2e3e5", title: key };
  return (
    <span
      className="data-label"
      title={style.title}
      style={{ backgroundColor: style.bg, color: style.fg }}
    >
      {String(kind).toUpperCase()}
    </span>
  );
}
