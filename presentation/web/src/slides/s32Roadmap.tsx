import { RoadmapColumn } from '../components/blocks';
import { RedCallout, SlideTitle } from '../components/core';
import { ROADMAP } from '../data/facts';

export function S32Roadmap() {
  return (
    <>
      <SlideTitle>Development roadmap</SlideTitle>
      <div className="grid flex-1 grid-cols-4 items-stretch gap-5">
        {ROADMAP.map((r) => (
          <RoadmapColumn key={r.q} q={r.q} title={r.title} items={r.items} now={r.now} />
        ))}
      </div>
      <p className="mt-3 font-mono text-[12px] text-muted">
        Timeline subject to revision based on clinical validation results and partner-site
        availability.
      </p>
      <RedCallout>
        The next two quarters are committed and specific; everything after is sequenced, not
        speculative.
      </RedCallout>
    </>
  );
}
