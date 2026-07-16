import { SHSTable } from '../components/blocks';
import { SoWhatBar, SlideTitle } from '../components/core';
import { GAP_TABLE } from '../data/facts';

export function S08Gap() {
  return (
    <>
      <SlideTitle>Six gaps — six answers</SlideTitle>
      <SHSTable
        head={['Current state', "What's needed", 'What we build']}
        rows={GAP_TABLE.map(([cur, need, built]) => [
          <span className="text-ink/75">{cur}</span>,
          need,
          <span className="font-semibold text-teal-ink">{built}</span>,
        ])}
      />
      <SoWhatBar>
        The gap is a generative, uncertainty-aware twin. We built it — the next act shows what
        exists today.
      </SoWhatBar>
    </>
  );
}
