import { RedCallout } from '../components/core';

/** Embeds the handover deck's densest system diagram unchanged — zero fact drift. */
export function S10Overview() {
  return (
    <>
      <div className="flex flex-1 items-center justify-center">
        <img
          src={`${import.meta.env.BASE_URL}assets/03-customer-twin-overview.svg`}
          alt="Customer Twin system overview: five data inputs feed three transformer models with uncertainty quantification, producing five output products"
          className="max-h-full w-full object-contain"
        />
      </div>
      <RedCallout label="White-box AI">
        Every prediction ships with uncertainty bounds and named, interpretable conditioning
        features — trust is built in, not bolted on.
      </RedCallout>
    </>
  );
}
