import { SHSTable } from '../components/blocks';
import { RedCallout, SlideTitle } from '../components/core';
import { TELEKOM_STUDY, THESIS } from '../data/facts';

const RQS = [
  ['RQ1', 'Does collaboration produce emergent diagnoses no single agent reaches?', `${THESIS.results.emergence} emergent`],
  ['RQ2', 'Does the multi-agent system beat the strongest single-agent baseline?', `88.3% vs 83.3% (+${THESIS.results.marginPp.toFixed(1)} pp)`],
  ['RQ3', 'What is the latency cost of orchestration?', `${THESIS.results.latencyMas} vs ${THESIS.results.latencyBaseline}`],
] as const;

export function A05ThesisEval() {
  return (
    <>
      <SlideTitle lede={`${THESIS.results.setup} · ${THESIS.llmTiers}`}>
        Appendix · thesis evaluation design
      </SlideTitle>
      <div className="grid flex-1 grid-cols-[1.05fr_1fr] items-start gap-8">
        <div className="flex flex-col gap-2.5">
          {RQS.map(([rq, q, a]) => (
            <div key={rq} className="flex items-center gap-4 rounded-lg border border-mist bg-paper px-5 py-3">
              <span className="font-mono text-[14px] font-bold text-teal-ink">{rq}</span>
              <span className="flex-1 text-[14.5px] leading-snug text-ink">{q}</span>
              <span className="kpi-number text-[15px] font-bold whitespace-nowrap text-navy">{a}</span>
            </div>
          ))}
          <div className="rounded-lg border-l-4 border-orange bg-orange/[0.06] px-5 py-3 text-[13.5px] leading-snug text-ink/85">
            <strong className="text-orange-ink">Results provenance:</strong> figures are the final
            evaluation run (results/ablation/thesis_summary.json, 2026-04-09), confirmed against
            thesis <strong>Chapter 5</strong>. The repository README table is an earlier, weaker
            run — do not quote it.
          </div>
        </div>
        <div className="flex flex-col gap-2.5">
          <div>
            <div className="font-mono text-[11px] tracking-[0.13em] text-navy uppercase">
              {TELEKOM_STUDY.title}
            </div>
            <div className="mt-0.5 font-mono text-[11px] text-muted">{TELEKOM_STUDY.date}</div>
          </div>
          <SHSTable head={TELEKOM_STUDY.head} rows={TELEKOM_STUDY.rows} mono={[1, 2, 3, 4]} className="text-[12px]" />
          <p className="rounded-md bg-surface px-4 py-2.5 text-[13px] leading-snug text-ink/85 italic">
            {TELEKOM_STUDY.lesson}
          </p>
          <p className="text-[12px] leading-snug text-muted">{TELEKOM_STUDY.note}</p>
        </div>
      </div>
      <RedCallout label="Appendix">
        Five modes per case isolate the collaboration effect — and the question was tested in two
        domains: Siemens MRI (left) and Telekom IoT (right).
      </RedCallout>
    </>
  );
}
