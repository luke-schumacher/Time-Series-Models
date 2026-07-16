import type { SlideDef } from './types';

export interface NavState {
  index: number;
  fragment: number;
}

const fragmentsOf = (slide: SlideDef | undefined): number => slide?.fragments ?? 0;

export function clampState(state: NavState, slides: SlideDef[]): NavState {
  const index = Math.max(0, Math.min(state.index, slides.length - 1));
  const fragment = Math.max(0, Math.min(state.fragment, fragmentsOf(slides[index])));
  return { index, fragment };
}

/** Advance one step: fragments first, then the next slide (fragment 0). */
export function next(state: NavState, slides: SlideDef[]): NavState {
  const s = clampState(state, slides);
  if (s.fragment < fragmentsOf(slides[s.index])) {
    return { index: s.index, fragment: s.fragment + 1 };
  }
  if (s.index < slides.length - 1) {
    return { index: s.index + 1, fragment: 0 };
  }
  return s;
}

/** Step back: fragments first, then the previous slide fully revealed. */
export function prev(state: NavState, slides: SlideDef[]): NavState {
  const s = clampState(state, slides);
  if (s.fragment > 0) {
    return { index: s.index, fragment: s.fragment - 1 };
  }
  if (s.index > 0) {
    return { index: s.index - 1, fragment: fragmentsOf(slides[s.index - 1]) };
  }
  return s;
}

/**
 * Parse "#/12" (1-based slide number) or "#/s14" / "#/a02" (slide id).
 * An optional ".f" suffix deep-links a fragment state, e.g. "#/34.4".
 */
export function parseHash(hash: string, slides: SlideDef[]): NavState | null {
  const m = hash.match(/^#\/([^/.]+)(?:\.(\d+))?$/);
  if (!m) return null;
  const [, token, frag] = m;
  const fragment = frag ? parseInt(frag, 10) : 0;
  if (/^\d+$/.test(token)) {
    const n = parseInt(token, 10);
    if (n >= 1 && n <= slides.length) return clampState({ index: n - 1, fragment }, slides);
    return null;
  }
  const idx = slides.findIndex((s) => s.id === token);
  return idx >= 0 ? clampState({ index: idx, fragment }, slides) : null;
}

export function formatHash(state: NavState): string {
  return `#/${state.index + 1}`;
}

/** Fraction [0,1] of the talk completed, for the schedule bar cursor. */
export function talkProgress(state: NavState, slides: SlideDef[]): number {
  const core = slides.filter((s) => !s.appendix);
  if (core.length === 0) return 0;
  const pos = Math.min(state.index, core.length - 1);
  if (slides[state.index]?.appendix) return 1;
  return pos / Math.max(1, core.length - 1);
}
