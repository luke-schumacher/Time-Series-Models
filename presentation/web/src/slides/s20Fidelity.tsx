import { MetricCard } from '../components/cards';
import { RedCallout, SlideTitle } from '../components/core';
import { DAY_FIDELITY as F } from '../data/facts';

export function S20Fidelity() {
  return (
    <>
      <SlideTitle lede="Held-out validation against real schedules — not training-set fit.">
        Day-level fidelity: the three numbers that matter
      </SlideTitle>
      <div className="grid flex-1 grid-cols-3 items-center gap-6">
        <MetricCard value={F.regionOrder} label="body-region order match" sub={F.regionOrderSub} accent="teal" />
        <MetricCard value={F.dayLength} label="total day duration" sub={F.dayLengthSub} accent="teal" />
        <MetricCard value={F.exchangeError} label="exchange duration error" sub={F.exchangeErrorSub} accent="teal" />
      </div>
      <RedCallout>
        Good enough to schedule against — day structure, day length and transition times all hold
        at fleet scale.
      </RedCallout>
    </>
  );
}
