import { DeckProvider, useDeck } from './deck/DeckContext';
import { Overview } from './deck/Overview';
import { PrintView } from './deck/PrintView';
import { SlideChrome } from './deck/SlideChrome';
import { SpeakerNotes } from './deck/SpeakerNotes';
import { Stage } from './deck/Stage';
import { ACTS } from './deck/acts';
import { SLIDES } from './slides';

function CurrentSlide() {
  const { slide, index } = useDeck();
  const Body = slide.element;
  return (
    <SlideChrome key={index}>
      <Body />
    </SlideChrome>
  );
}

function KeyHint() {
  const { index } = useDeck();
  if (index !== 0) return null;
  return (
    <div className="absolute bottom-6 left-8 z-20 font-mono text-[12px] tracking-wide text-white/45">
      → advance · Esc overview · N notes · F fullscreen
    </div>
  );
}

export function App() {
  if (new URLSearchParams(window.location.search).has('print')) {
    return <PrintView slides={SLIDES} acts={ACTS} />;
  }
  return (
    <DeckProvider slides={SLIDES} acts={ACTS}>
      <Stage>
        <CurrentSlide />
        <KeyHint />
        <SpeakerNotes />
        <Overview />
      </Stage>
    </DeckProvider>
  );
}
