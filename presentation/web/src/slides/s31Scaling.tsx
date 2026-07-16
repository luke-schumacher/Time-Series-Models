import { CompareColumns } from '../components/blocks';
import { SoWhatBar, SlideTitle } from '../components/core';
import { SCALING } from '../data/facts';

/** Backbone → site-adapter schematic (concept only — no invented numbers). */
function ScalingDiagram() {
  const sites = ['Site A adapter', 'Site B adapter', 'Site N adapter'];
  return (
    <svg viewBox="0 0 1200 120" className="mb-5 w-full" aria-label="Shared backbone feeding lightweight per-site adapter heads">
      <rect x="30" y="30" width="330" height="60" rx="8" fill="#003087" />
      <text x="195" y="55" textAnchor="middle" fontSize="17" fontWeight="700" fill="#fff">
        Shared backbone
      </text>
      <text x="195" y="76" textAnchor="middle" fontSize="12" fill="#CCEEEE">
        pre-trained across all sites
      </text>
      {sites.map((s, i) => {
        const y = 14 + i * 36;
        return (
          <g key={s}>
            <path d={`M 360 60 C 430 60, 450 ${y + 16}, 520 ${y + 16}`} fill="none" stroke="#009999" strokeWidth="2" />
            <rect x="520" y={y} width="210" height="32" rx="6" fill="#009999" opacity={0.14} stroke="#009999" strokeWidth="1.4" />
            <text x="625" y={y + 21} textAnchor="middle" fontSize="13.5" fontWeight="600" fill="#007070">
              {s}
            </text>
          </g>
        );
      })}
      <text x="770" y="42" fontSize="13" fill="#6B7B7B" fontFamily="IBM Plex Mono, monospace">
        + customer-ID embedding as conditioning
      </text>
      <text x="770" y="66" fontSize="13" fill="#6B7B7B" fontFamily="IBM Plex Mono, monospace">
        fine-tune on ≤ 4 weeks of local data
      </text>
      <text x="770" y="90" fontSize="13" fill="#6B7B7B" fontFamily="IBM Plex Mono, monospace">
        cold start &lt; 2 h on CPU (~500 days history)
      </text>
    </svg>
  );
}

export function S31Scaling() {
  return (
    <>
      <SlideTitle lede="From 40 known scanners to any customer site.">
        Scaling architecture: single customer → fleet
      </SlideTitle>
      <ScalingDiagram />
      <CompareColumns
        left={{ title: 'Current — single-customer architecture', items: SCALING.current }}
        right={{ title: 'Target — multi-site architecture', items: SCALING.target }}
      />
      <SoWhatBar>{SCALING.insight}</SoWhatBar>
    </>
  );
}
