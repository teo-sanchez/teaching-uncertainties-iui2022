import { Component, Stream } from '@marcellejs/core';
import View from './slider-chart.view.svelte';
import { VerticalSliderOptions, SliderColourOptions } from '../../types';

export interface SliderChartData {
  x: string,
  y: number,
  colour?: SliderColourOptions 
};

export class SliderChart extends Component {
  title = 'slider-chart [custom module 🤖]';

  $options: Stream<VerticalSliderOptions>;
  $data: Stream<Array<SliderChartData>>;


  constructor(dataStream: Stream<Array<SliderChartData>>,
  options?: Partial<VerticalSliderOptions>) {
    super();
    this.$options = new Stream({
      min: 0,
      max: 1,
      float:  true,
      springValues: {
        stiffness: 0.15,
        damping: 0.4,
      },
      ...options,
    } as VerticalSliderOptions, true);
    this.$options.start();
    this.$data = dataStream;
  }

  mount(target?: HTMLElement): void {
    const t = target || document.querySelector(`#${this.id}`);
    if (!t) return;
    this.destroy();
    this.$$.app = new View({
      target: t,
      props: {
        title: this.title,
        data: this.$data,
        options: this.$options,
      },
    });
  }
}
