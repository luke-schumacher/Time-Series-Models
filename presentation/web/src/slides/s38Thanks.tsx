import { PRESENTER } from '../data/facts';

export function S38Thanks() {
  return (
    <div className="relative h-full w-full bg-navy-deep">
      <div className="absolute top-0 bottom-0 left-0 w-[10px] bg-teal" />
      <svg className="absolute right-0 bottom-0" width="420" height="380" viewBox="0 0 420 380" aria-hidden>
        <circle cx="310" cy="290" r="90" fill="none" stroke="#009999" strokeWidth="1.6" opacity="0.35" />
        <circle cx="310" cy="290" r="142" fill="none" stroke="#009999" strokeWidth="1" opacity="0.2" />
      </svg>
      <div className="relative flex h-full flex-col justify-center px-28">
        <h1 className="font-display text-[68px] leading-tight font-extrabold text-white">
          Thank you.
        </h1>
        <p className="mt-4 max-w-[820px] text-[22px] leading-snug text-teal">
          Questions welcome — the appendix behind this slide holds the full architecture, the
          run-book, and the evaluation designs.
        </p>
        <div className="mt-12 flex items-center gap-5 text-[16px] text-white/70">
          <span className="font-semibold text-white">{PRESENTER}</span>
          <span className="text-teal">·</span>
          <span>Customer Twin Project Team</span>
          <span className="text-teal">·</span>
          <span>Siemens Healthineers · AI &amp; Digital Health</span>
        </div>
        <p className="mt-8 font-mono text-[13px] text-white/40">
          press → for appendix · Esc for overview
        </p>
      </div>
    </div>
  );
}
