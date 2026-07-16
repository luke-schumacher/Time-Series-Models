import { PROBLEM } from '../data/facts';
import { CHART } from './chartTheme';

const W = 660;
const ROW_H = 52;
const LABEL_W = 170;
const PLOT_W = W - LABEL_W - 112; // reserve room for the "15 – 35 min" end labels
const X_MAX = 35;

const x = (min: number) => LABEL_W + (min / X_MAX) * PLOT_W;

/**
 * S05: per-region min–max scan-duration ranges — the deck's table, made
 * visceral. Identity is carried by the row labels; values are direct-labeled.
 */
export function DurationRanges() {
  const rows = PROBLEM.durationRanges;
  const H = rows.length * ROW_H + 56;
  const spread = Math.round(
    Math.max(...rows.map((r) => r.max)) / Math.min(...rows.map((r) => r.min)),
  ); // 35 / 0.5 = 70

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
      {/* grid + axis */}
      {[0, 5, 10, 15, 20, 25, 30, 35].map((v) => (
        <g key={v}>
          <line x1={x(v)} y1={10} x2={x(v)} y2={H - 42} stroke={CHART.grid} strokeWidth={1} />
          <text x={x(v)} y={H - 26} textAnchor="middle" fontSize={11.5} fill={CHART.axis} fontFamily="IBM Plex Mono, monospace">
            {v}
          </text>
        </g>
      ))}
      <text x={LABEL_W + PLOT_W / 2} y={H - 6} textAnchor="middle" fontSize={12.5} fill={CHART.label}>
        Typical scan duration (minutes) — one scanner serves all of these
      </text>

      {rows.map((r, i) => {
        const y = 22 + i * ROW_H;
        const x1 = x(r.min);
        const x2 = x(r.max);
        return (
          <g key={r.region}>
            <text x={LABEL_W - 12} y={y + 13} textAnchor="end" fontSize={13.5} fontWeight={600} fill={CHART.ink}>
              {r.region}
            </text>
            {/* track */}
            <rect x={x(0)} y={y + 4} width={PLOT_W} height={10} rx={5} fill={CHART.grid} opacity={0.55} />
            {/* range */}
            <rect x={x1} y={y} width={Math.max(x2 - x1, 6)} height={18} rx={6} fill={CHART.teal} />
            <text x={x2 + 10} y={y + 13.5} fontSize={11.5} fill="#007070" fontWeight={600} fontFamily="IBM Plex Mono, monospace">
              {r.range}
            </text>
          </g>
        );
      })}

      {/* spread annotation — derived from the table itself (0.5 → 35 min) */}
      <text x={W - 6} y={16} textAnchor="end" fontSize={12.5} fontWeight={700} fill="#B04D00" fontFamily="IBM Plex Mono, monospace">
        {spread}× spread
      </text>
    </svg>
  );
}
