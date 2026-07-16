import { describe, expect, it } from 'vitest';
import { ACTS } from '../deck/acts';
import { SLIDES } from './index';

describe('slide registry', () => {
  it('has 38 core slides and 6 appendix slides', () => {
    expect(SLIDES.filter((s) => !s.appendix)).toHaveLength(38);
    expect(SLIDES.filter((s) => s.appendix)).toHaveLength(6);
  });

  it('has unique ids', () => {
    const ids = SLIDES.map((s) => s.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it('gives every slide a title and speaker notes', () => {
    for (const s of SLIDES) {
      expect(s.title.trim(), s.id).not.toBe('');
      expect(s.notes.trim().length, `${s.id} notes`).toBeGreaterThan(20);
    }
  });

  it('orders slides by act, appendix last', () => {
    const acts = SLIDES.map((s) => s.act);
    expect([...acts].sort((a, b) => a - b)).toEqual(acts);
    const firstAppendix = SLIDES.findIndex((s) => s.appendix);
    expect(SLIDES.slice(firstAppendix).every((s) => s.appendix && s.act === 8)).toBe(true);
  });

  it('references only defined acts, and covers every act', () => {
    const actIds = new Set(ACTS.map((a) => a.id));
    for (const s of SLIDES) expect(actIds.has(s.act), `${s.id} act ${s.act}`).toBe(true);
    for (const a of ACTS) {
      expect(SLIDES.some((s) => s.act === a.id), `act ${a.id} has slides`).toBe(true);
    }
  });

  it('keeps fragment counts sane', () => {
    for (const s of SLIDES) {
      const f = s.fragments ?? 0;
      expect(f, `${s.id} fragments`).toBeGreaterThanOrEqual(0);
      expect(f, `${s.id} fragments`).toBeLessThanOrEqual(6);
    }
  });

  it('keeps act time budgets summing to the 45-minute talk', () => {
    expect(ACTS.reduce((sum, a) => sum + a.minutes, 0)).toBe(45);
  });
});
