import { PhaseCard } from '../components/cards';
import { Pill, RedCallout, SlideTitle } from '../components/core';
import { JOURNEY } from '../data/facts';
import { useFragment } from '../deck/DeckContext';

export function S09Journey() {
  const visible = [true, useFragment(1), useFragment(2), useFragment(3)];
  return (
    <>
      <SlideTitle lede="Four deliberate phases — each one de-risked the next.">
        The three-year journey
      </SlideTitle>
      <div className="grid flex-1 grid-cols-4 items-stretch gap-5">
        {JOURNEY.map((p, i) => (
          <div key={p.phase} className="frag" data-visible={visible[i]}>
            <PhaseCard
              n={i + 1}
              phase={p.phase}
              desc={p.desc}
              outcome={p.outcome}
              chip={<Pill tone="navy">{p.year}</Pill>}
            />
          </div>
        ))}
      </div>
      <RedCallout>
        No phase was skipped — and behind each one, many model variants were tried, evaluated and
        retired. The version history is the audit trail.
      </RedCallout>
    </>
  );
}
