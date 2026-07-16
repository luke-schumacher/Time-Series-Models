import { SHSTable } from '../components/blocks';
import { RedCallout, SlideTitle } from '../components/core';
import { RUNBOOK } from '../data/facts';

export function A02Runbook() {
  return (
    <>
      <SlideTitle lede="The full local pipeline — the Databricks pipeline runs the same steps via spark_pipeline.py.">
        Appendix · run_all.py — step by step
      </SlideTitle>
      <SHSTable
        head={['Step', 'Action', 'Output']}
        rows={RUNBOOK.map(([s, a, o]) => [
          <strong className="text-navy">{s}</strong>,
          a,
          <span className="text-muted">{o}</span>,
        ])}
        mono={[0, 2]}
        className="text-[13px]"
      />
      <RedCallout label="Handover">
        One entry point, seven stages, resumable per step — a new owner can run the whole system
        from this table.
      </RedCallout>
    </>
  );
}
