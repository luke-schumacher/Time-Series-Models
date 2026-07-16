import { RedCallout, SlideTitle } from '../components/core';
import { ArchStepThrough } from '../viz/ArchStepThrough';

export function S15Architecture() {
  return (
    <>
      <SlideTitle lede="One unified design shared by the Exchange and Examination models — three tiers, each with one job.">
        Inside the model
      </SlideTitle>
      <ArchStepThrough />
      <RedCallout label="Depth on demand">
        This is the mental model — the full architecture diagram with every feature named is in the
        appendix.
      </RedCallout>
    </>
  );
}
