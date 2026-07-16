import { MetricCard } from '../components/cards';
import { RedCallout, SlideTitle } from '../components/core';
import { HEADLINE } from '../data/facts';

export function S02Numbers() {
  return (
    <>
      <SlideTitle lede="The complete story of this report, before any of the story.">
        Three years in five numbers
      </SlideTitle>
      <div className="grid flex-1 grid-cols-5 content-center gap-5">
        {HEADLINE.map((h, i) => (
          <MetricCard
            key={h.label}
            value={h.value}
            unit={h.unit}
            label={h.label}
            sub={h.sub}
            accent={i === 3 ? 'orange' : i === 4 ? 'navy' : 'teal'}
          />
        ))}
      </div>
      <RedCallout>
        Every number on this slide is measured, not promised — the next 45 minutes is the story
        behind each one.
      </RedCallout>
    </>
  );
}
