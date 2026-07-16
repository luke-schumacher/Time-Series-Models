import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react';
import type { ActDef, SlideDef } from './types';
import { clampState, formatHash, next, parseHash, prev, type NavState } from './logic';

interface DeckApi {
  slides: SlideDef[];
  acts: ActDef[];
  index: number;
  fragment: number;
  slide: SlideDef;
  overview: boolean;
  notesOpen: boolean;
  startedAt: number | null;
  goTo: (index: number) => void;
  setOverview: (v: boolean) => void;
  setNotesOpen: (v: boolean) => void;
}

const DeckContext = createContext<DeckApi | null>(null);

export function useDeck(): DeckApi {
  const ctx = useContext(DeckContext);
  if (!ctx) throw new Error('useDeck outside DeckProvider');
  return ctx;
}

/** Convenience for slides: is fragment step `at` revealed yet? */
export function useFragment(at: number): boolean {
  return useDeck().fragment >= at;
}

/** Static provider for the print view: one slide, all fragments revealed. */
export function StaticDeckProvider({
  slides,
  acts,
  index,
  children,
}: {
  slides: SlideDef[];
  acts: ActDef[];
  index: number;
  children: ReactNode;
}) {
  const noop = () => {};
  const value: DeckApi = {
    slides,
    acts,
    index,
    fragment: slides[index].fragments ?? 0,
    slide: slides[index],
    overview: false,
    notesOpen: false,
    startedAt: null,
    goTo: noop,
    setOverview: noop,
    setNotesOpen: noop,
  };
  return <DeckContext.Provider value={value}>{children}</DeckContext.Provider>;
}

export function DeckProvider({
  slides,
  acts,
  children,
}: {
  slides: SlideDef[];
  acts: ActDef[];
  children: ReactNode;
}) {
  const [state, setState] = useState<NavState>(() => {
    return parseHash(window.location.hash, slides) ?? { index: 0, fragment: 0 };
  });
  const [overview, setOverview] = useState(false);
  const [notesOpen, setNotesOpen] = useState(false);
  const [startedAt, setStartedAt] = useState<number | null>(null);
  const startedRef = useRef(false);

  const markStarted = useCallback(() => {
    if (!startedRef.current) {
      startedRef.current = true;
      setStartedAt(Date.now());
    }
  }, []);

  const goTo = useCallback(
    (index: number) => {
      markStarted();
      setState(clampState({ index, fragment: 0 }, slides));
      setOverview(false);
    },
    [slides, markStarted],
  );

  // hash ← state
  useEffect(() => {
    const target = formatHash(state);
    if (window.location.hash !== target) {
      window.history.replaceState(null, '', target);
    }
  }, [state]);

  // state ← hash (deep links / manual edits)
  useEffect(() => {
    const onHash = () => {
      const parsed = parseHash(window.location.hash, slides);
      if (parsed) {
        setState((cur) =>
          parsed.index === cur.index && parsed.fragment === cur.fragment ? cur : parsed,
        );
      }
    };
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, [slides]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA') return;
      switch (e.code) {
        case 'ArrowRight':
        case 'Space':
        case 'PageDown':
          e.preventDefault();
          markStarted();
          setState((s) => next(s, slides));
          break;
        case 'ArrowLeft':
        case 'PageUp':
          e.preventDefault();
          markStarted();
          setState((s) => prev(s, slides));
          break;
        case 'Home':
          e.preventDefault();
          setState({ index: 0, fragment: 0 });
          break;
        case 'End':
          e.preventDefault();
          setState({ index: slides.length - 1, fragment: 0 });
          break;
        case 'Escape':
        case 'KeyO':
          e.preventDefault();
          setOverview((v) => !v);
          break;
        case 'KeyN':
          e.preventDefault();
          setNotesOpen((v) => !v);
          break;
        case 'KeyF':
          e.preventDefault();
          if (document.fullscreenElement) void document.exitFullscreen();
          else void document.documentElement.requestFullscreen();
          break;
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [slides, markStarted]);

  const value = useMemo<DeckApi>(
    () => ({
      slides,
      acts,
      index: state.index,
      fragment: state.fragment,
      slide: slides[state.index],
      overview,
      notesOpen,
      startedAt,
      goTo,
      setOverview,
      setNotesOpen,
    }),
    [slides, acts, state, overview, notesOpen, startedAt, goTo],
  );

  return <DeckContext.Provider value={value}>{children}</DeckContext.Provider>;
}
