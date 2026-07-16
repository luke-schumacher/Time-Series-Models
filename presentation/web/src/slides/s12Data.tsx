import { StatTile } from '../components/cards';
import { RedCallout, SlideTitle } from '../components/core';
import { DATA_FOUNDATION as D } from '../data/facts';

export function S12Data() {
  return (
    <>
      <SlideTitle lede="Everything downstream is learned from real fleet behaviour — no synthetic training data.">
        The data foundation
      </SlideTitle>
      <div className="grid grid-cols-5 gap-5">
        <StatTile value={String(D.scanners)} label="MRI scanners" sub={`serials ${D.serials.split(' · ')[0]} …`} />
        <StatTile value={D.windowLen} label="of raw event logs" sub={D.window} />
        <StatTile value={String(D.vocab)} label="source-ID vocabulary" sub={D.vocabDesc} />
        <StatTile value={String(D.regions)} label="body-region classes" sub={D.regionsDesc} />
        <StatTile value={String(D.seqTypes)} label="pulse-sequence types" sub={D.seqTypesDesc} />
      </div>
      <div className="mt-6 rounded-lg bg-surface px-6 py-4">
        <div className="font-mono text-[11.5px] tracking-[0.13em] text-teal-ink uppercase">
          scanner serials
        </div>
        <div className="mt-1 font-mono text-[15px] text-ink/80">{D.serials}</div>
      </div>
      <RedCallout>{D.eventsNote}</RedCallout>
    </>
  );
}
