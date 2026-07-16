import { RedCallout } from '../components/core';

export function A01Transformer() {
  return (
    <>
      <div className="flex flex-1 items-center justify-center">
        <img
          src={`${import.meta.env.BASE_URL}assets/05-transformer-architecture.svg`}
          alt="Full unified transformer architecture: Tier 1 conditioning encoder with patient, temporal and coil features; Tier 2 autoregressive token decoder; Tier 3 bidirectional duration head"
          className="max-h-full w-full object-contain"
        />
      </div>
      <RedCallout label="Appendix">
        The unabridged three-tier architecture — every feature group, mask, and loss term as
        implemented.
      </RedCallout>
    </>
  );
}
