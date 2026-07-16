import { ORCHESTRATOR } from '../data/facts';

const chipBase =
  'grid h-[46px] place-items-center rounded-md font-mono text-[13px] font-semibold tracking-wide';

/** S22: predicted body-region order vs ground truth — one day, one scanner. */
export function SequenceCompare() {
  const { truth, predicted, mismatchIndex } = ORCHESTRATOR;

  return (
    <div className="flex flex-col gap-0">
      <div className="grid grid-cols-[130px_1fr] items-center gap-4">
        <span className="text-right text-[14.5px] font-bold text-navy">Ground truth</span>
        <div className="grid grid-cols-6 gap-2">
          {truth.map((t, i) => (
            <div
              key={i}
              className={`${chipBase} ${
                t === 'BREAK' ? 'bg-surface text-muted ring-1 ring-mist' : 'bg-navy text-white'
              }`}
            >
              {t}
            </div>
          ))}
        </div>
      </div>

      {/* connectors */}
      <div className="grid grid-cols-[130px_1fr] gap-4">
        <span />
        <div className="grid h-[34px] grid-cols-6 gap-2">
          {truth.map((_, i) => (
            <div key={i} className="relative">
              {i === mismatchIndex ? (
                <>
                  <div className="absolute left-1/2 h-full w-0 -translate-x-1/2 border-l-2 border-dashed border-orange" />
                  <span className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-paper px-1 font-mono text-[13px] font-bold text-orange-ink">
                    ≠
                  </span>
                </>
              ) : (
                <div className="absolute left-1/2 h-full w-0 -translate-x-1/2 border-l-2 border-teal/60" />
              )}
            </div>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-[130px_1fr] items-center gap-4">
        <span className="text-right text-[14.5px] font-bold text-teal-ink">Predicted</span>
        <div className="grid grid-cols-6 gap-2">
          {predicted.map((t, i) => (
            <div
              key={i}
              className={`${chipBase} ${
                i === mismatchIndex
                  ? 'bg-orange/10 text-orange-ink ring-2 ring-orange'
                  : t === 'BREAK'
                    ? 'bg-surface text-muted ring-1 ring-mist'
                    : 'bg-teal/12 text-teal-ink ring-1 ring-teal/50'
              }`}
            >
              {t}
            </div>
          ))}
        </div>
      </div>

      <div className="mt-3 grid grid-cols-[130px_1fr] gap-4">
        <span />
        <p className="font-mono text-[13px] text-muted">{ORCHESTRATOR.exampleNote}</p>
      </div>
    </div>
  );
}
