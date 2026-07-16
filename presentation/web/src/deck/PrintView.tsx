import { StaticDeckProvider } from './DeckContext';
import { SlideChrome } from './SlideChrome';
import { STAGE_H, STAGE_W } from './Stage';
import type { ActDef, SlideDef } from './types';

const PRINT_SCALE = 0.66; // 1600×900 → 1056×594, fits A4 landscape

/** `?print` — every slide fully revealed, one per page. */
export function PrintView({ slides, acts }: { slides: SlideDef[]; acts: ActDef[] }) {
  return (
    <div className="print-root bg-white">
      {slides.map((s, i) => {
        const Body = s.element;
        return (
          <div
            key={s.id}
            className="print-page"
            style={{ width: STAGE_W * PRINT_SCALE, height: STAGE_H * PRINT_SCALE }}
          >
            <div
              className="relative origin-top-left overflow-hidden bg-paper"
              style={{ width: STAGE_W, height: STAGE_H, transform: `scale(${PRINT_SCALE})` }}
            >
              <StaticDeckProvider slides={slides} acts={acts} index={i}>
                <SlideChrome>
                  <Body />
                </SlideChrome>
              </StaticDeckProvider>
            </div>
          </div>
        );
      })}
    </div>
  );
}
