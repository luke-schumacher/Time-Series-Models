import { useState } from 'react';
import { RedCallout, SlideTitle } from '../components/core';
import { ROI } from '../data/facts';
import { useFragment } from '../deck/DeckContext';
import { RoiWaterfall } from '../viz/RoiWaterfall';

/** n machines × $17.7K/yr — pure arithmetic on the stated per-machine figure. */
function FleetSlider() {
  const [fleet, setFleet] = useState<number>(ROI.fleetDefault);
  const totalK = fleet * ROI.perMachineK;
  const total = totalK >= 1000 ? `$${(totalK / 1000).toFixed(2)}M` : `$${Math.round(totalK)}K`;

  return (
    <div className="flex items-center gap-6 rounded-lg bg-surface px-6 py-3.5">
      <label className="flex items-center gap-3 font-mono text-[13px] font-semibold whitespace-nowrap text-ink">
        fleet size
        <input
          type="range"
          min={1}
          max={100}
          step={1}
          value={fleet}
          onChange={(e) => setFleet(Number(e.target.value))}
          className="w-56 accent-teal"
          aria-label="Number of MRI machines in the fleet"
        />
        <span className="w-24 text-teal-ink">
          {fleet} machine{fleet === 1 ? '' : 's'}
        </span>
      </label>
      <div className="h-8 w-px bg-mist" />
      <div className="text-[16px]">
        <span className="text-muted">× {ROI.recoverable}/yr each ≈ </span>
        <span className="kpi-number text-[24px] font-bold text-orange-ink">{total}</span>
        <span className="font-semibold text-ink"> recoverable per year</span>
        <span className="ml-3 font-mono text-[12px] text-muted">
          {ROI.fleetDefault} = today's modelled fleet
        </span>
      </div>
    </div>
  );
}

export function S34Roi() {
  const showFleet = useFragment(4);
  return (
    <>
      <SlideTitle lede="From a quantified annual loss to what the twin recovers — step by step.">
        Financial impact per MRI machine, per year
      </SlideTitle>
      <div className="flex flex-1 flex-col justify-center">
        <RoiWaterfall />
      </div>
      <div className="frag" data-visible={showFleet}>
        <FleetSlider />
      </div>
      <p className="mt-2 font-mono text-[11.5px] text-muted">{ROI.disclaimer}</p>
      <RedCallout>
        {ROI.recoverablePct} of a known, quantified loss is recoverable — {ROI.recoverable} per
        machine, per year, before coil-simulation and VR upside.
      </RedCallout>
    </>
  );
}
