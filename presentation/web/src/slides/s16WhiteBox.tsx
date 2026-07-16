import { CompareColumns } from '../components/blocks';
import { RedCallout, SlideTitle } from '../components/core';
import { WHITEBOX } from '../data/facts';

export function S16WhiteBox() {
  return (
    <>
      <SlideTitle lede="In a clinical environment, explainability is not a nice-to-have — it is the product.">
        White-box by construction
      </SlideTitle>
      <CompareColumns
        left={{ title: 'Typical black-box approach', items: WHITEBOX.black }}
        right={{ title: 'Our white-box architecture', items: WHITEBOX.white }}
      />
      <RedCallout>
        Clinicians and engineers can understand every output — and each model can be tested,
        validated, and improved independently.
      </RedCallout>
    </>
  );
}
