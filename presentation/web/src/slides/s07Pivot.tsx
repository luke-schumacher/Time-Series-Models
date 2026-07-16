import { CompareColumns } from '../components/blocks';
import { RedCallout, SlideTitle } from '../components/core';

export function S07Pivot() {
  return (
    <>
      <SlideTitle>From reporting to simulation</SlideTitle>
      <CompareColumns
        left={{
          title: 'Today — static reporting',
          items: [
            'Manual scheduling on fixed slot durations',
            'Post-hoc reports on downtime and utilisation',
            'Reactive maintenance after hardware failure',
            'No forward-looking capacity simulation',
            'Coil / hardware ROI assessed only after purchase',
          ],
        }}
        right={{
          title: 'Tomorrow — generative simulation',
          items: [
            <><strong>Generative AI</strong> produces synthetic daily schedules</>,
            <>Real-time uncertainty bounds <strong>μ ± σ</strong> per scan</>,
            <>Predictive maintenance — <strong>2 weeks</strong> advance warning</>,
            'Pre-purchase ROI modelling for hardware upgrades',
            'VR training without occupying a live scanner',
          ],
        }}
      />
      <RedCallout>
        This transition is the strategic mandate of the Four Twins programme — the right column is
        what the rest of this deck demonstrates.
      </RedCallout>
    </>
  );
}
