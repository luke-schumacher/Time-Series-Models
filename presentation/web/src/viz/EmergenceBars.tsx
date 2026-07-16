import { THESIS } from '../data/facts';
import { CHART } from './chartTheme';

const W = 1120;
const ROW_H = 52;
const LABEL_W = 250;
const BAR_MAX = W - LABEL_W - 190;

const KIND = {
  single: { fill: CHART.faint, tag: 'single agent' },
  baseline: { fill: CHART.navy, tag: 'strongest baseline — one agent, all data' },
  mas: { fill: CHART.teal, tag: 'ours — three agents + synthesis' },
} as const;

/** S27: keyword accuracy across the five evaluation modes (12 fault cases each). */
export function EmergenceBars() {
  const modes = THESIS.results.modes;
  const H = modes.length * ROW_H + 40;
  const baseline = modes.find((m) => m.kind === 'baseline')!;
  const mas = modes.find((m) => m.kind === 'mas')!;
  const bx = (v: number) => LABEL_W + (v / 100) * BAR_MAX;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
      {[0, 25, 50, 75, 100].map((v) => (
        <g key={v}>
          <line x1={bx(v)} y1={8} x2={bx(v)} y2={H - 32} stroke={CHART.grid} strokeWidth={1} />
          <text x={bx(v)} y={H - 14} textAnchor="middle" fontSize={12} fill={CHART.axis} fontFamily="IBM Plex Mono, monospace">
            {v}%
          </text>
        </g>
      ))}

      {modes.map((m, i) => {
        const k = KIND[m.kind as keyof typeof KIND];
        const yTop = 14 + i * ROW_H;
        const isMas = m.kind === 'mas';
        return (
          <g key={m.mode}>
            <text x={LABEL_W - 14} y={yTop + 15} textAnchor="end" fontSize={14.5} fontWeight={isMas ? 800 : 600} fill={isMas ? '#007070' : CHART.ink}>
              {m.mode}
            </text>
            <text x={LABEL_W - 14} y={yTop + 32} textAnchor="end" fontSize={11} fill={CHART.axis}>
              {k.tag}
            </text>
            <rect x={LABEL_W} y={yTop} width={bx(m.acc) - LABEL_W} height={34} rx={4} fill={k.fill} />
            <text x={bx(m.acc) + 12} y={yTop + 22} fontSize={16} fontWeight={800} fill={isMas ? '#007070' : CHART.ink} fontFamily="Archivo Variable, sans-serif">
              {m.acc.toFixed(1)}%
            </text>
          </g>
        );
      })}

      {/* +5.0 pp bracket between baseline and MAS */}
      {(() => {
        const yB = 14 + 3 * ROW_H + 17;
        const yM = 14 + 4 * ROW_H + 17;
        const xEnd = Math.max(bx(baseline.acc), bx(mas.acc)) + 92;
        return (
          <g>
            <path
              d={`M ${bx(baseline.acc) + 74} ${yB} H ${xEnd} V ${yM} H ${bx(mas.acc) + 78}`}
              fill="none"
              stroke={CHART.orange}
              strokeWidth={2}
              strokeDasharray="5 4"
            />
            <text x={xEnd + 10} y={(yB + yM) / 2 + 5} fontSize={17} fontWeight={800} fill="#B04D00" fontFamily="Archivo Variable, sans-serif">
              +{THESIS.results.marginPp.toFixed(1)} pp
            </text>
          </g>
        );
      })()}
    </svg>
  );
}
