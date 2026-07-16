import { useFragment } from '../deck/DeckContext';
import { ROI } from '../data/facts';
import { CHART } from './chartTheme';

const W = 1340; // wide enough that the recoverable bracket never clips
const H = 430;
const PLOT_X = 96;
const PLOT_TOP = 30;
const PLOT_H = 320;
const BAR_W = 132;
const GAP = 58;
const MAX = 25;
const GRID_END = 1040; // gridlines stop at the bars; the bracket zone stays clean

const y = (v: number) => PLOT_TOP + (1 - v / MAX) * PLOT_H;
const yBase = y(0);

/** S34: $25K baseline loss → three recoveries → $7.3K residual. Fragment-stepped. */
export function RoiWaterfall() {
  // fragment 0: baseline · 1..3: recovery steps · 4: residual + recoverable banner
  const visibleSteps = [useFragment(1), useFragment(2), useFragment(3)];
  const showResidual = useFragment(4);

  let cum = ROI.baseline;
  const stepRects = ROI.steps.map((s, i) => {
    const top = cum;
    cum += s.delta; // deltas are negative
    return { ...s, top, bottom: cum, visible: visibleSteps[i] };
  });

  const bx = (i: number) => PLOT_X + 40 + i * (BAR_W + GAP);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
      {/* y axis */}
      {[0, 5, 10, 15, 20, 25].map((v) => (
        <g key={v}>
          <line x1={PLOT_X} y1={y(v)} x2={GRID_END} y2={y(v)} stroke={CHART.grid} strokeWidth={1} />
          <text x={PLOT_X - 10} y={y(v) + 4} textAnchor="end" fontSize={12.5} fill={CHART.axis} fontFamily="IBM Plex Mono, monospace">
            ${v}K
          </text>
        </g>
      ))}
      <text x={26} y={PLOT_TOP + PLOT_H / 2} fontSize={13} fill={CHART.label} transform={`rotate(-90 26 ${PLOT_TOP + PLOT_H / 2})`} textAnchor="middle">
        Annual value per machine (USD)
      </text>

      {/* baseline bar */}
      <rect x={bx(0)} y={y(ROI.baseline)} width={BAR_W} height={yBase - y(ROI.baseline)} rx={4} fill={CHART.axis} />
      <text x={bx(0) + BAR_W / 2} y={y(ROI.baseline) - 12} textAnchor="middle" fontSize={17} fontWeight={700} fill={CHART.ink}>
        ${ROI.baseline.toFixed(0)}K
      </text>
      <text x={bx(0) + BAR_W / 2} y={yBase + 22} textAnchor="middle" fontSize={13.5} fontWeight={600} fill={CHART.ink}>
        Baseline
      </text>
      <text x={bx(0) + BAR_W / 2} y={yBase + 40} textAnchor="middle" fontSize={12} fill={CHART.axis}>
        annual loss
      </text>

      {/* recovery steps */}
      {stepRects.map((s, i) => {
        const px = bx(i + 1);
        return (
          <g key={s.label} className="frag" data-visible={s.visible}>
            <line x1={bx(i) + BAR_W} y1={y(s.top)} x2={px} y2={y(s.top)} stroke={CHART.axis} strokeDasharray="4 4" strokeWidth={1.2} />
            <rect x={px} y={y(s.top)} width={BAR_W} height={y(s.bottom) - y(s.top)} rx={4} fill={CHART.teal} />
            <text x={px + BAR_W / 2} y={y(s.top) - 12} textAnchor="middle" fontSize={17} fontWeight={700} fill="#007070">
              −${Math.abs(s.delta)}K
            </text>
            <text x={px + BAR_W / 2} y={yBase + 22} textAnchor="middle" fontSize={13.5} fontWeight={600} fill={CHART.ink}>
              {s.label}
            </text>
            <text x={px + BAR_W / 2} y={yBase + 40} textAnchor="middle" fontSize={12} fill={CHART.axis}>
              {s.sub}
            </text>
          </g>
        );
      })}

      {/* residual + recoverable bracket */}
      <g className="frag" data-visible={showResidual}>
        <line x1={bx(3) + BAR_W} y1={y(ROI.residual)} x2={bx(4)} y2={y(ROI.residual)} stroke={CHART.axis} strokeDasharray="4 4" strokeWidth={1.2} />
        <rect x={bx(4)} y={y(ROI.residual)} width={BAR_W} height={yBase - y(ROI.residual)} rx={4} fill={CHART.navy} />
        <text x={bx(4) + BAR_W / 2} y={y(ROI.residual) - 12} textAnchor="middle" fontSize={17} fontWeight={700} fill="#003087">
          ${ROI.residual}K
        </text>
        <text x={bx(4) + BAR_W / 2} y={yBase + 22} textAnchor="middle" fontSize={13.5} fontWeight={600} fill={CHART.ink}>
          Net cost
        </text>
        <text x={bx(4) + BAR_W / 2} y={yBase + 40} textAnchor="middle" fontSize={12} fill={CHART.axis}>
          after digital twin
        </text>

        {(() => {
          const brX = bx(4) + BAR_W + 46;
          return (
            <g>
              <path
                d={`M ${brX} ${y(ROI.baseline)} h 14 V ${y(ROI.residual)} h -14`}
                fill="none"
                stroke={CHART.orange}
                strokeWidth={2.5}
              />
              <text x={brX + 34} y={(y(ROI.baseline) + y(ROI.residual)) / 2 - 26} fontSize={14} fill={CHART.label}>
                Total recoverable
              </text>
              <text x={brX + 34} y={(y(ROI.baseline) + y(ROI.residual)) / 2 + 6} fontSize={26} fontWeight={800} fill="#B04D00" fontFamily="Archivo Variable, sans-serif">
                {ROI.recoverable}
              </text>
              <text x={brX + 34} y={(y(ROI.baseline) + y(ROI.residual)) / 2 + 30} fontSize={13.5} fill={CHART.label}>
                per machine / year ({ROI.recoverablePct})
              </text>
            </g>
          );
        })()}
      </g>
    </svg>
  );
}
