import { useMemo, useState } from 'react';
import { UNCERTAINTY } from '../data/facts';
import { CHART } from './chartTheme';

const W = 1120;
const H = 330;
const PLOT_X = 76;
const PLOT_W = W - PLOT_X - 30;
const PLOT_TOP = 26;
const PLOT_H = 240;
const X_MAX = 32;
const Y_MAX = 0.36; // HEAD peak ≈ 0.332

const px = (min: number) => PLOT_X + (min / X_MAX) * PLOT_W;
const py = (d: number) => PLOT_TOP + (1 - d / Y_MAX) * PLOT_H;
const yBase = py(0);

const pdf = (xv: number, mu: number, sigma: number) =>
  Math.exp(-((xv - mu) ** 2) / (2 * sigma ** 2)) / (sigma * Math.sqrt(2 * Math.PI));

function curvePath(mu: number, sigma: number, close: boolean, from = 0, to = X_MAX): string {
  const pts: string[] = [];
  for (let m = from; m <= to + 1e-9; m += 0.2) {
    pts.push(`${px(m).toFixed(1)},${py(pdf(m, mu, sigma)).toFixed(1)}`);
  }
  const line = `M ${pts.join(' L ')}`;
  return close ? `${line} L ${px(to).toFixed(1)},${yBase} L ${px(from).toFixed(1)},${yBase} Z` : line;
}

/** S17: what μ±σ buys the schedule — slider converts σ into concrete slack. */
export function UncertaintyExplorer() {
  const [k, setK] = useState(1.0);
  const { head, spine } = UNCERTAINTY;

  const paths = useMemo(
    () => ({
      headLine: curvePath(head.mu, head.sigma, false),
      headBand: curvePath(head.mu, head.sigma, true, head.mu - k * head.sigma, head.mu + k * head.sigma),
      spineLine: curvePath(spine.mu, spine.sigma, false),
      spineBand: curvePath(spine.mu, spine.sigma, true, spine.mu - k * spine.sigma, spine.mu + k * spine.sigma),
    }),
    [k, head, spine],
  );

  const headSlack = (k * head.sigma).toFixed(1);
  const spineSlack = (k * spine.sigma).toFixed(1);

  return (
    <div className="flex flex-col gap-1">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
        {[0, 5, 10, 15, 20, 25, 30].map((v) => (
          <g key={v}>
            <line x1={px(v)} y1={PLOT_TOP} x2={px(v)} y2={yBase} stroke={CHART.grid} strokeWidth={1} />
            <text x={px(v)} y={yBase + 20} textAnchor="middle" fontSize={12.5} fill={CHART.axis} fontFamily="IBM Plex Mono, monospace">
              {v}
            </text>
          </g>
        ))}
        <text x={px(X_MAX / 2)} y={yBase + 44} textAnchor="middle" fontSize={13.5} fill={CHART.label}>
          Predicted scan duration (minutes)
        </text>
        <text x={30} y={PLOT_TOP + PLOT_H / 2} fontSize={12.5} fill={CHART.axis} transform={`rotate(-90 30 ${PLOT_TOP + PLOT_H / 2})`} textAnchor="middle">
          probability density
        </text>
        <line x1={PLOT_X} y1={yBase} x2={PLOT_X + PLOT_W} y2={yBase} stroke={CHART.axis} strokeWidth={1.2} />

        {/* ±kσ bands */}
        <path d={paths.headBand} fill={CHART.navy} opacity={0.18} />
        <path d={paths.spineBand} fill={CHART.teal} opacity={0.2} />
        {/* curves */}
        <path d={paths.headLine} fill="none" stroke={CHART.navy} strokeWidth={2.5} />
        <path d={paths.spineLine} fill="none" stroke={CHART.teal} strokeWidth={2.5} />

        {/* direct labels */}
        <text x={px(head.mu)} y={py(pdf(head.mu, head.mu, head.sigma)) - 12} textAnchor="middle" fontSize={14.5} fontWeight={700} fill="#003087">
          HEAD · μ {head.mu.toFixed(0)} min · σ {head.sigma} min
        </text>
        <text x={px(spine.mu) + 46} y={py(pdf(spine.mu, spine.mu, spine.sigma)) - 14} textAnchor="middle" fontSize={14.5} fontWeight={700} fill="#007070">
          SPINE · μ {spine.mu.toFixed(0)} min · σ {spine.sigma.toFixed(0)} min
        </text>
      </svg>

      <div className="flex items-center gap-6 rounded-lg bg-surface px-6 py-4">
        <label className="flex items-center gap-3 font-mono text-[13px] font-semibold whitespace-nowrap text-ink">
          slack policy
          <input
            type="range"
            min={0.5}
            max={2}
            step={0.05}
            value={k}
            onChange={(e) => setK(Number(e.target.value))}
            className="w-56 accent-teal"
            aria-label="Slack multiplier k in units of sigma"
          />
          <span className="w-14 text-teal-ink">±{k.toFixed(2)}σ</span>
        </label>
        <div className="h-8 w-px bg-mist" />
        <div className="flex gap-8 text-[15px]">
          <span>
            <span className="font-semibold text-navy">HEAD slot:</span>{' '}
            <span className="kpi-number font-bold text-navy">{head.mu} ± {headSlack} min</span>
          </span>
          <span>
            <span className="font-semibold text-teal-ink">SPINE slot:</span>{' '}
            <span className="kpi-number font-bold text-teal-ink">{spine.mu} ± {spineSlack} min</span>
          </span>
          <span className="text-muted">
            same policy, ~{(spine.sigma / head.sigma).toFixed(1)}× the buffer — allocated where variance actually lives
          </span>
        </div>
      </div>
    </div>
  );
}
