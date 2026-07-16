import { PRESENTED, PRESENTER } from '../data/facts';

export function S01Title() {
  return (
    <div className="relative h-full w-full bg-navy-deep">
      {/* identity edge */}
      <div className="absolute top-0 bottom-0 left-0 w-[10px] bg-teal" />
      {/* MRI bore motif — concentric circles, upper right (ported from the deck) */}
      <svg className="absolute top-0 right-0" width="460" height="420" viewBox="0 0 460 420" aria-hidden>
        <circle cx="330" cy="120" r="96" fill="none" stroke="#009999" strokeWidth="1.6" opacity="0.35" />
        <circle cx="330" cy="120" r="150" fill="none" stroke="#009999" strokeWidth="1" opacity="0.2" />
        <circle cx="330" cy="120" r="208" fill="none" stroke="#009999" strokeWidth="0.8" opacity="0.12" />
      </svg>

      <div className="relative flex h-full flex-col justify-center px-28">
        <div className="font-mono text-[14px] font-medium tracking-[0.22em] text-teal uppercase">
          Three-year results report
        </div>
        <h1 className="font-display mt-5 max-w-[1100px] text-[76px] leading-[1.02] font-extrabold text-white">
          MRI Digital Twin
        </h1>
        <p className="mt-5 max-w-[880px] text-[24px] leading-snug text-teal">
          Generative simulation &amp; semantic intelligence for Siemens Healthineers
        </p>

        <div className="mt-14 flex flex-col gap-2 text-[16px] text-white/70">
          <div className="flex items-center gap-5">
            <span className="font-semibold text-white">{PRESENTER}</span>
            <span className="text-teal">·</span>
            <span>{PRESENTED}</span>
          </div>
          <div className="flex items-center gap-5 text-[14.5px] text-white/55">
            <span>Customer Twin Project Team</span>
            <span className="text-teal">·</span>
            <span>Siemens Healthineers</span>
          </div>
        </div>
      </div>
    </div>
  );
}
