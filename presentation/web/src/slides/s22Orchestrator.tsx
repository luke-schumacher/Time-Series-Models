import { RedCallout, SlideTitle } from '../components/core';
import { ORCHESTRATOR } from '../data/facts';
import { SequenceCompare } from '../viz/SequenceCompare';

export function S22Orchestrator() {
  return (
    <>
      <SlideTitle lede={ORCHESTRATOR.validation}>
        Orchestrator validation — writing the day autonomously
      </SlideTitle>
      <div className="flex flex-1 flex-col justify-center gap-7">
        <SequenceCompare />
        <div className="grid grid-cols-2 gap-6">
          <div className="flex items-baseline gap-4 rounded-lg bg-surface px-6 py-4">
            <span className="kpi-number text-[34px] font-bold text-navy">{ORCHESTRATOR.editDistance}</span>
            <span className="text-[15px] leading-tight text-ink/80">
              mean sequence edit distance across the validation set
            </span>
          </div>
          <div className="flex items-baseline gap-4 rounded-lg bg-surface px-6 py-4">
            <span className="kpi-number text-[34px] font-bold text-teal-ink">{ORCHESTRATOR.breakAccuracy}</span>
            <span className="text-[15px] leading-tight text-ink/80">
              BREAK-token placement accuracy (lunch &amp; handover gaps)
            </span>
          </div>
        </div>
      </div>
      <RedCallout>
        No ground-truth patient list exists at inference time — the orchestrator writes the day's
        body-region order itself, including breaks.
      </RedCallout>
    </>
  );
}
