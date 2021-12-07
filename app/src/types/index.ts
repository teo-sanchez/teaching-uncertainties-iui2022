
import { ClassifierResults, SliderOptions } from '@marcellejs/core';

export interface Uncertainty extends Record<string, number> {
  aleatoric: number;
  epistemic: number; 
}

export interface EnsembleResults extends ClassifierResults {
  perClassConfidences: Record<string, Array<number>>,
  perClassMeanConfidences: Record<string, number>,
  perModelConfidences: Array<Record<string, number>>
  perModelLabels: Array<string>;
  // + label and confidences from ClassifierResults
};


export interface DensityEstimatorResults {
  distance: number,
  [key: string]: any
  // + label and confidences from ClassifierResults
};
export interface VerticalSliderOptions extends Partial<SliderOptions> {
  // see options for slider Svelte component etc.
  // here: https://simeydotme.github.io/svelte-range-slider-pips/
  min: number;
  max: number;
  float: boolean;
  springValues: {
    stiffness: number;
    damping: number;
  },
  colour: SliderColourOptions,
  step: number;
}

export interface SliderColourOptions {
  main: string, // string matching a valid css colour value
  focussed: string,
}