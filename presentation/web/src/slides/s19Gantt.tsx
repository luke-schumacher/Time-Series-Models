import { RedCallout, SlideTitle } from '../components/core';
import { GanttComparison } from '../viz/GanttComparison';

export function S19Gantt() {
  return (
    <>
      <SlideTitle lede="The structure of a generated day against what actually happened.">
        A simulated day vs reality
      </SlideTitle>
      <div className="flex flex-1 flex-col justify-center">
        <GanttComparison />
      </div>
      <RedCallout>
        Same scanner, same date — body-region order, block lengths and exchange gaps are all
        reproduced by the model, token by token.
      </RedCallout>
    </>
  );
}
