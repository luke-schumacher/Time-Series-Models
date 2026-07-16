import { ScreenshotSlot } from '../components/blocks';
import { SoWhatBar, SlideTitle } from '../components/core';
import { THESIS } from '../data/facts';

export function S28ThesisDemo() {
  return (
    <>
      <SlideTitle lede="A working system, not a benchmark harness — brought up with one command.">
        What we can show live
      </SlideTitle>
      <div className="grid flex-1 grid-cols-[1.25fr_1fr] items-stretch gap-8">
        <ScreenshotSlot
          name="agentic-chat-ui.png"
          hint="Chat interface mid-diagnosis: delegation badges lighting up per agent, reasoning-chain trace expanded, sources panel on the right."
          className="!p-3"
        >
          <div className="flex h-full w-full flex-col gap-2">
            <img
              src={`${import.meta.env.BASE_URL}assets/screenshots/agentic-chat-ui.png`}
              alt="Agentic Infra Co-Pilot chat interface: agent status chips for Governance, Hardware and Telemetry, MAGNETOM-specific example prompts, and a scanner-context bar"
              className="min-h-0 w-full flex-1 rounded-md object-contain"
            />
            <p className="text-center font-mono text-[11px] text-muted">
              live capture (idle) — swap for a mid-diagnosis shot to show delegation badges
            </p>
          </div>
        </ScreenshotSlot>
        <div className="flex flex-col justify-center gap-3.5">
          {THESIS.demo.map((d) => (
            <div key={d} className="flex gap-3 rounded-lg border border-mist bg-paper px-5 py-3.5 text-[16px] leading-snug text-ink/90">
              <span className="mt-[8px] h-1.5 w-4 shrink-0 rounded bg-teal" />
              {d}
            </div>
          ))}
          <p className="mt-1 text-[14px] leading-snug text-muted">
            Live demo or screenshots — both work; the system runs locally via docker-compose with
            no external dependencies beyond the LLM APIs.
          </p>
        </div>
      </div>
      <SoWhatBar>
        The twin predicts the day; the copilot explains the fleet — two proofs that this data
        estate supports production AI.
      </SoWhatBar>
    </>
  );
}
