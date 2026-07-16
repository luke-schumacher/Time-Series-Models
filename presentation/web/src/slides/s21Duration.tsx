import { SHSTable } from '../components/blocks';
import { RedCallout, SlideTitle } from '../components/core';
import { DURATION_PRED } from '../data/facts';
import { RegionBars } from '../viz/RegionBars';

export function S21Duration() {
  return (
    <>
      <SlideTitle lede="Predicted μ against observed median duration, per body region.">
        Duration prediction — mean and spread
      </SlideTitle>
      <div className="grid flex-1 grid-cols-[1.25fr_1fr] items-center gap-10">
        <RegionBars />
        <div>
          <SHSTable head={['Metric', 'Value']} rows={DURATION_PRED.metrics} mono={[1]} />
          <p className="mt-4 rounded-lg border border-teal/40 bg-teal/[0.05] px-4 py-3 text-[14.5px] leading-snug text-ink/85">
            σ calibration within 20% means the error bars themselves are trustworthy — the model
            knows what it doesn't know.
          </p>
        </div>
      </div>
      <RedCallout>{DURATION_PRED.interpretation}</RedCallout>
    </>
  );
}
