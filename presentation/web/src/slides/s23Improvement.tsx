import { Pill, RedCallout, SlideTitle } from '../components/core';
import { IMPROVEMENT_PROVENANCE } from '../data/facts';
import { ImprovementTimeline } from '../viz/ImprovementTimeline';

export function S23Improvement() {
  return (
    <>
      <SlideTitle
        lede={
          <>
            The pipeline did not start this good — three diagnosed root causes, three validated
            fixes. <Pill tone="teal">{IMPROVEMENT_PROVENANCE}</Pill>
          </>
        }
      >
        How the model improved
      </SlideTitle>
      <ImprovementTimeline />
      <RedCallout>
        Every stage had a measurable exit gate — and passed it. That is what "validated" means in
        this deck.
      </RedCallout>
    </>
  );
}
