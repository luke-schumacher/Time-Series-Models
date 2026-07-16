import { SHSTable } from '../components/blocks';
import { SoWhatBar, SlideTitle } from '../components/core';
import { DECISIONS } from '../data/facts';

export function S18Decisions() {
  return (
    <>
      <SlideTitle lede="Each of these was validated the hard way — by watching training fail without it.">
        Hard-won architecture decisions
      </SlideTitle>
      <SHSTable
        head={['Decision', 'Why', 'Evidence']}
        rows={DECISIONS.map((d) => [
          <strong className="text-navy">{d.decision}</strong>,
          d.why,
          <span className="text-teal-ink">{d.evidence}</span>,
        ])}
      />
      <SoWhatBar>
        These scars are why the results in the next act hold — none of this design is speculative.
      </SoWhatBar>
    </>
  );
}
