import { MetricCard } from '../components/cards';
import { RedCallout, SlideTitle } from '../components/core';
import { PROBLEM } from '../data/facts';

export function S06HiddenCost() {
  return (
    <>
      <SlideTitle lede="What unpredictability costs — and what digital twins have already recovered elsewhere.">
        The hidden cost of scheduling inefficiency
      </SlideTitle>
      <div className="grid flex-1 grid-cols-4 content-center gap-5">
        <MetricCard
          value="$25K"
          label="lost per machine, per year"
          sub={`Unplanned downtime + scheduling inefficiency — ${PROBLEM.baselineLossSource.toLowerCase()}`}
          accent="orange"
        />
        <MetricCard
          value="−44.8%"
          label="patient wait time"
          sub={`Digital-twin dynamic waitlist management — ${PROBLEM.literatureSource}`}
          accent="teal"
        />
        <MetricCard
          value="+14.5%"
          label="machine utilization"
          sub={`Reinforcement learning in digital-twin environments — ${PROBLEM.literatureSource}`}
          accent="teal"
        />
        <MetricCard
          value="2 wks"
          label="advance failure warning"
          sub="Tube-life + arcing-event monitoring — crisis becomes a plan"
          accent="navy"
        />
      </div>
      <RedCallout>
        The recovery is already proven in the literature — the open question was an MRI-grade twin
        to deliver it. That is what we built.
      </RedCallout>
    </>
  );
}
