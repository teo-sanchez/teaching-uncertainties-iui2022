import { Stream } from '@marcellejs/core';
import { VerticalSliderOptions } from '../../types';
import { SliderChart, SliderChartData } from './slider-chart.component';

export function sliderChart(
  dataStream: Stream<Array<SliderChartData>>,
  options?: Partial<VerticalSliderOptions>): SliderChart {
  return new SliderChart(dataStream, options);
}

export type { SliderChart, SliderChartData, VerticalSliderOptions as SliderChartOptions};
