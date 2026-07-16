import { useRef, useState } from 'react';
import { DURATION_PRED } from '../data/facts';
import { CHART } from './chartTheme';

const W = 640;
const H = 400;
const PLOT_X = 62;
const PLOT_TOP = 46;
const PLOT_H = 280;
const BAR_W = 36;
const IN_GAP = 5;
const MAX = 25;

const y = (v: number) => PLOT_TOP + (1 - v / MAX) * PLOT_H;
const yBase = y(0);

/** S21: observed vs predicted μ per body region (grouped bars, hover detail). */
export function RegionBars() {
  const [tip, setTip] = useState<{ left: number; top: number; lines: string[] } | null>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const groupW = (W - PLOT_X - 30) / DURATION_PRED.regions.length;

  const hover = (i: number | null, e?: React.MouseEvent) => {
    if (i === null || !e || !wrapRef.current) {
      setTip(null);
      return;
    }
    const r = wrapRef.current.getBoundingClientRect();
    const obs = DURATION_PRED.observed[i];
    const pred = DURATION_PRED.predicted[i];
    setTip({
      left: ((e.clientX - r.left) / r.width) * 100,
      top: ((e.clientY - r.top) / r.height) * 100,
      lines: [
        `${DURATION_PRED.regions[i]}`,
        `observed ${obs.toFixed(1)} min · predicted μ ${pred.toFixed(1)} min`,
        `Δ ${(pred - obs) >= 0 ? '+' : ''}${(pred - obs).toFixed(1)} min`,
      ],
    });
  };

  return (
    <div ref={wrapRef} className="relative">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
        {/* legend */}
        <g transform={`translate(${PLOT_X}, 10)`}>
          <rect width={13} height={13} rx={2.5} fill={CHART.navy} />
          <text x={19} y={11} fontSize={13} fill={CHART.label}>Observed median</text>
          <rect x={172} width={13} height={13} rx={2.5} fill={CHART.teal} />
          <text x={191} y={11} fontSize={13} fill={CHART.label}>Predicted μ</text>
        </g>

        {[0, 5, 10, 15, 20, 25].map((v) => (
          <g key={v}>
            <line x1={PLOT_X} y1={y(v)} x2={W - 20} y2={y(v)} stroke={CHART.grid} strokeWidth={1} />
            <text x={PLOT_X - 9} y={y(v) + 4} textAnchor="end" fontSize={12} fill={CHART.axis} fontFamily="IBM Plex Mono, monospace">
              {v}
            </text>
          </g>
        ))}
        <text x={20} y={PLOT_TOP + PLOT_H / 2} fontSize={12.5} fill={CHART.axis} transform={`rotate(-90 20 ${PLOT_TOP + PLOT_H / 2})`} textAnchor="middle">
          Duration (min)
        </text>

        {DURATION_PRED.regions.map((region, i) => {
          const cx = PLOT_X + i * groupW + groupW / 2;
          const obs = DURATION_PRED.observed[i];
          const pred = DURATION_PRED.predicted[i];
          return (
            <g key={region} onMouseMove={(e) => hover(i, e)} onMouseLeave={() => hover(null)}>
              {/* invisible hit target wider than the marks */}
              <rect x={cx - groupW / 2} y={PLOT_TOP} width={groupW} height={PLOT_H + 26} fill="transparent" />
              <rect x={cx - BAR_W - IN_GAP / 2} y={y(obs)} width={BAR_W} height={yBase - y(obs)} rx={4} fill={CHART.navy} />
              <rect x={cx + IN_GAP / 2} y={y(pred)} width={BAR_W} height={yBase - y(pred)} rx={4} fill={CHART.teal} />
              <text x={cx} y={yBase + 20} textAnchor="middle" fontSize={13} fontWeight={600} fill={CHART.ink}>
                {region}
              </text>
            </g>
          );
        })}
        <line x1={PLOT_X} y1={yBase} x2={W - 20} y2={yBase} stroke={CHART.axis} strokeWidth={1.2} />
      </svg>

      {tip && (
        <div
          className="pointer-events-none absolute z-10 -translate-x-1/2 -translate-y-[115%] rounded-md bg-navy-deep px-3 py-2 font-mono text-[11.5px] leading-relaxed whitespace-nowrap text-white shadow-lg"
          style={{ left: `${tip.left}%`, top: `${tip.top}%` }}
        >
          {tip.lines.map((l, i) => (
            <div key={i} className={i === 0 ? 'font-bold text-teal' : ''}>{l}</div>
          ))}
        </div>
      )}
    </div>
  );
}
