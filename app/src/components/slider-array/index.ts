import { SliderArray } from './slider-array.component';
import { VerticalSliderOptions } from '../../types';

export function sliderArray(
    items: string[],
    options: Partial<VerticalSliderOptions>[]): SliderArray {
  return new SliderArray(items, options);
}

export type { SliderArray, VerticalSliderOptions as SliderArrayOptions };
