import { RedCallout, SlideTitle } from '../components/core';
import { THESIS } from '../data/facts';

export function S26ThesisArch() {
  return (
    <>
      <SlideTitle lede={THESIS.stack}>Three specialists, one diagnosis</SlideTitle>

      <div className="flex flex-1 flex-col justify-center gap-4">
        <div className="grid grid-cols-3 gap-5">
          {THESIS.agents.map((a) => (
            <div key={a.name} className="rounded-lg border-t-4 border-mist border-t-teal bg-paper p-5 shadow-sm">
              <h3 className="font-display text-[21px] font-bold text-navy">{a.name} Agent</h3>
              <p className="mt-1 text-[14.5px] text-ink/80">{a.scope}</p>
              <p className="mt-3 font-mono text-[13px] text-teal-ink">
                <span className="kpi-number text-[22px] font-bold">{a.docs.toLocaleString('en-US')}</span>{' '}
                docs · own RAG store
              </p>
            </div>
          ))}
        </div>

        <div className="mx-auto h-5 w-px bg-mist" />

        <div className="flex items-center justify-center gap-2.5 rounded-lg bg-surface px-6 py-3.5">
          <span className="mr-2 font-mono text-[11.5px] tracking-[0.13em] text-navy uppercase">
            Autonomy protocol
          </span>
          {THESIS.protocol.map((p) => (
            <span key={p} className="rounded border border-teal/40 bg-paper px-2.5 py-1 font-mono text-[12px] font-semibold text-teal-ink">
              {p}
            </span>
          ))}
        </div>

        <div className="mx-auto h-5 w-px bg-mist" />

        <div className="mx-auto flex w-fit items-center gap-5 rounded-lg bg-navy px-8 py-4 text-white shadow-md">
          <span className="font-display text-[20px] font-bold">Synthesizer</span>
          <span className="text-teal">→</span>
          <span className="text-[16px]">one fused diagnosis, with sources and risk level</span>
        </div>

        <p className="text-center font-mono text-[12.5px] text-muted">{THESIS.llmTiers}</p>
      </div>

      <RedCallout>
        An agent can answer, consult a peer, redirect, or refuse — "not my domain" is the protocol
        working, not a failure.
      </RedCallout>
    </>
  );
}
