import { useEffect, useRef, useState } from 'react';
import { GANTT } from '../data/facts';
import { CHART, minutesToClock, REGION_FILL } from './chartTheme';

const W = 1180;
const H = 292;
const PLOT_X = 118;
const PLOT_W = W - PLOT_X - 24;
const LANE_H = 46;
const LANE_REAL_Y = 52;
const LANE_SIM_Y = 138;
const AXIS_Y = 214;

const x = (m: number) => PLOT_X + (m / GANTT.total) * PLOT_W;

interface Block {
  readonly region: string;
  readonly from: number;
  readonly to: number;
}

function Lane({
  blocks,
  y,
  onHover,
}: {
  blocks: readonly Block[];
  y: number;
  onHover: (b: Block | null, e?: React.MouseEvent) => void;
}) {
  return (
    <g>
      {blocks.map((b, i) => {
        const bx = x(b.from);
        const bw = x(b.to) - bx;
        return (
          <g key={i}>
            <rect
              x={bx + 1}
              y={y}
              width={Math.max(bw - 2, 2)}
              height={LANE_H}
              rx={3}
              fill={REGION_FILL[b.region]}
              onMouseMove={(e) => onHover(b, e)}
              onMouseLeave={() => onHover(null)}
            />
            {bw > 56 && (
              <text
                x={bx + 9}
                y={y + LANE_H / 2 + 4.5}
                fill="#fff"
                fontSize={13}
                fontWeight={600}
                pointerEvents="none"
              >
                {b.region}
              </text>
            )}
          </g>
        );
      })}
    </g>
  );
}

/**
 * S19 signature visual: ground-truth vs simulated day, same scanner, same date.
 * Play sweeps a scan line across the simulated lane, assembling it in time order.
 */
export function GanttComparison() {
  const [progress, setProgress] = useState(1); // 0..1 of the sim lane revealed
  const [playing, setPlaying] = useState(false);
  const [tip, setTip] = useState<{ left: number; top: number; text: string } | null>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const raf = useRef(0);

  useEffect(() => () => cancelAnimationFrame(raf.current), []);

  const play = () => {
    cancelAnimationFrame(raf.current);
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduced) {
      setProgress(1);
      return;
    }
    setPlaying(true);
    const t0 = performance.now();
    const dur = 3600;
    const step = (t: number) => {
      const p = Math.min((t - t0) / dur, 1);
      setProgress(p);
      if (p < 1) raf.current = requestAnimationFrame(step);
      else setPlaying(false);
    };
    setProgress(0);
    raf.current = requestAnimationFrame(step);
  };

  const hover = (b: Block | null, e?: React.MouseEvent) => {
    if (!b || !e || !wrapRef.current) {
      setTip(null);
      return;
    }
    const r = wrapRef.current.getBoundingClientRect();
    setTip({
      left: ((e.clientX - r.left) / r.width) * 100,
      top: ((e.clientY - r.top) / r.height) * 100,
      text: `${b.region === 'EXCH' ? 'Exchange' : b.region} · ${minutesToClock(b.from)}–${minutesToClock(b.to)} · ${b.to - b.from} min`,
    });
  };

  const clipW = Math.max(0, progress * (PLOT_W + 4));
  const scanX = PLOT_X + progress * PLOT_W;

  return (
    <div ref={wrapRef} className="relative">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
        {/* lane titles */}
        <text x={PLOT_X - 12} y={LANE_REAL_Y + LANE_H / 2 + 5} textAnchor="end" fontSize={14.5} fontWeight={700} fill="#003087">
          Ground truth
        </text>
        <text x={PLOT_X - 12} y={LANE_SIM_Y + LANE_H / 2 + 5} textAnchor="end" fontSize={14.5} fontWeight={700} fill={CHART.teal}>
          Simulated
        </text>

        {/* grid + time axis */}
        {GANTT.ticks.map((t, i) => {
          const gx = PLOT_X + (i / (GANTT.ticks.length - 1)) * PLOT_W;
          return (
            <g key={t}>
              <line x1={gx} y1={38} x2={gx} y2={AXIS_Y - 12} stroke={CHART.grid} strokeWidth={1} />
              <text x={gx} y={AXIS_Y + 8} textAnchor="middle" fontSize={12.5} fill={CHART.axis} fontFamily="IBM Plex Mono, monospace">
                {t}
              </text>
            </g>
          );
        })}
        <text x={PLOT_X + PLOT_W / 2} y={AXIS_Y + 32} textAnchor="middle" fontSize={13} fill={CHART.label}>
          Time of day — same scanner, same date
        </text>

        <Lane blocks={GANTT.real} y={LANE_REAL_Y} onHover={hover} />

        <clipPath id="sim-clip">
          <rect x={PLOT_X - 2} y={LANE_SIM_Y - 6} width={clipW} height={LANE_H + 12} />
        </clipPath>
        <g clipPath="url(#sim-clip)">
          <Lane blocks={GANTT.sim} y={LANE_SIM_Y} onHover={hover} />
        </g>

        {playing && (
          <g>
            <line x1={scanX} y1={LANE_SIM_Y - 8} x2={scanX} y2={LANE_SIM_Y + LANE_H + 8} stroke={CHART.teal} strokeWidth={2.5} />
            <line x1={scanX} y1={LANE_SIM_Y - 8} x2={scanX} y2={LANE_SIM_Y + LANE_H + 8} stroke={CHART.teal} strokeWidth={7} opacity={0.25} />
          </g>
        )}

        {/* legend: exchange slivers are too narrow for direct labels */}
        <g transform={`translate(${PLOT_X}, ${AXIS_Y + 50})`}>
          <rect width={13} height={13} rx={2.5} fill={REGION_FILL.EXCH} />
          <text x={19} y={11} fontSize={12.5} fill={CHART.label}>
            Exchange — patient transition & coil change
          </text>
          <rect x={330} width={13} height={13} rx={2.5} fill={REGION_FILL.HEAD} />
          <text x={349} y={11} fontSize={12.5} fill={CHART.label}>
            Examination blocks (labeled per body region)
          </text>
        </g>
      </svg>

      <button
        onClick={play}
        disabled={playing}
        className="absolute top-0 right-0 flex items-center gap-2 rounded-md border border-teal/50 bg-teal/[0.08] px-3.5 py-1.5 font-mono text-[12.5px] font-semibold text-teal-ink transition-colors hover:bg-teal/15 disabled:opacity-50"
      >
        <svg width="11" height="12" viewBox="0 0 11 12" aria-hidden>
          <path d="M0 0l11 6-11 6z" fill="currentColor" />
        </svg>
        {playing ? 'generating…' : 'replay generation'}
      </button>

      {tip && (
        <div
          className="pointer-events-none absolute z-10 -translate-x-1/2 -translate-y-[130%] rounded-md bg-navy-deep px-3 py-1.5 font-mono text-[12px] whitespace-nowrap text-white shadow-lg"
          style={{ left: `${tip.left}%`, top: `${tip.top}%` }}
        >
          {tip.text}
        </div>
      )}
    </div>
  );
}
