import { describe, expect, it } from 'vitest';
import { clampState, formatHash, next, parseHash, prev, talkProgress } from './logic';
import type { SlideDef } from './types';

const el = () => null;
const mk = (id: string, fragments = 0, appendix = false): SlideDef => ({
  id,
  title: id,
  act: 0,
  fragments,
  notes: '',
  appendix,
  element: el,
});

const slides: SlideDef[] = [mk('s01'), mk('s02', 2), mk('s03'), mk('a01', 1, true)];

describe('navigation', () => {
  it('advances through fragments before slides', () => {
    let s = { index: 1, fragment: 0 };
    s = next(s, slides);
    expect(s).toEqual({ index: 1, fragment: 1 });
    s = next(s, slides);
    expect(s).toEqual({ index: 1, fragment: 2 });
    s = next(s, slides);
    expect(s).toEqual({ index: 2, fragment: 0 });
  });

  it('steps back to the previous slide fully revealed', () => {
    expect(prev({ index: 2, fragment: 0 }, slides)).toEqual({ index: 1, fragment: 2 });
    expect(prev({ index: 1, fragment: 2 }, slides)).toEqual({ index: 1, fragment: 1 });
  });

  it('clamps at both ends', () => {
    expect(prev({ index: 0, fragment: 0 }, slides)).toEqual({ index: 0, fragment: 0 });
    expect(next({ index: 3, fragment: 1 }, slides)).toEqual({ index: 3, fragment: 1 });
    expect(clampState({ index: 99, fragment: 99 }, slides)).toEqual({ index: 3, fragment: 1 });
  });
});

describe('hash routing', () => {
  it('round-trips numeric hashes (1-based)', () => {
    expect(formatHash({ index: 0, fragment: 0 })).toBe('#/1');
    expect(parseHash('#/3', slides)).toEqual({ index: 2, fragment: 0 });
  });

  it('resolves slide ids and rejects junk', () => {
    expect(parseHash('#/s02', slides)).toEqual({ index: 1, fragment: 0 });
    expect(parseHash('#/a01', slides)).toEqual({ index: 3, fragment: 0 });
    expect(parseHash('#/99', slides)).toBeNull();
    expect(parseHash('#/nope', slides)).toBeNull();
    expect(parseHash('', slides)).toBeNull();
  });

  it('deep-links fragment states with a ".f" suffix, clamped to the slide', () => {
    expect(parseHash('#/2.1', slides)).toEqual({ index: 1, fragment: 1 });
    expect(parseHash('#/s02.2', slides)).toEqual({ index: 1, fragment: 2 });
    expect(parseHash('#/2.9', slides)).toEqual({ index: 1, fragment: 2 });
    expect(parseHash('#/1.1', slides)).toEqual({ index: 0, fragment: 0 });
    expect(parseHash('#/2.x', slides)).toBeNull();
  });
});

describe('talk progress', () => {
  it('spans 0→1 over core slides and pins appendix at 1', () => {
    expect(talkProgress({ index: 0, fragment: 0 }, slides)).toBe(0);
    expect(talkProgress({ index: 2, fragment: 0 }, slides)).toBe(1);
    expect(talkProgress({ index: 3, fragment: 0 }, slides)).toBe(1);
  });
});
