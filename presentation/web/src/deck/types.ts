import type { ComponentType } from 'react';

export interface ActDef {
  /** Act number as shown on slides (0-based acts, appendix = 8) */
  id: number;
  label: string;
  /** Short label for the schedule bar / overview */
  short: string;
  /** Talk-time budget in minutes (appendix = 0, excluded from the bar) */
  minutes: number;
}

export interface SlideDef {
  /** Stable id, e.g. "s14" or "a02" — used in tests and the overview */
  id: string;
  /** Title shown in the overview grid (slides render their own headings) */
  title: string;
  /** Act id this slide belongs to */
  act: number;
  /** Number of fragment steps beyond the initial state (default 0) */
  fragments?: number;
  /** Speaker notes (N panel) */
  notes: string;
  /** 'hero' slides suppress the standard chrome (title/thanks) */
  chrome?: 'default' | 'hero';
  appendix?: boolean;
  element: ComponentType;
}
