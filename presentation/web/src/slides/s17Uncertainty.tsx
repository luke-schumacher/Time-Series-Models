import { RedCallout, SlideTitle } from '../components/core';
import { UNCERTAINTY } from '../data/facts';
import { UncertaintyExplorer } from '../viz/UncertaintyExplorer';

export function S17Uncertainty() {
  return (
    <>
      <SlideTitle lede={UNCERTAINTY.note}>Uncertainty that means something</SlideTitle>
      <UncertaintyExplorer />
      <RedCallout>
        High σ is not model weakness — it is honest scheduling guidance: buffer goes where variance
        actually lives.
      </RedCallout>
    </>
  );
}
