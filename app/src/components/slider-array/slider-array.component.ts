import { Component, Stream } from '@marcellejs/core';
import View from './slider-array.view.svelte';
import { VerticalSliderOptions } from '../../types';


export class SliderArray extends Component {
  title = 'slider-array [custom module 🤖]';

  $options: Stream<VerticalSliderOptions[]>;
  $items: Stream<string[]>;
  $values: Stream<number[]>;

  constructor(
    items: string[],
    options: Partial<VerticalSliderOptions>[]) {
    super();
    this.$options = new Stream(items.map((_, idx) => ({
      min: 0,
      max: 1,
      float:  true,
      springValues: {
        stiffness: 0.15,
        damping: 0.4,
      },
      ...options[idx],
    } as VerticalSliderOptions)), true);
    this.$options.start();

    this.$items = new Stream(items, true);
    this.$items.start();

    this.$values = new Stream([], true);
    this.$values.start();
    this.$items.subscribe(its => {
      const values = this.$values.value;
      const options = this.$options.value; 
      // const previousLength = values.length;
      values.length = its.length;
      options.length = its.length;
      //
      // values.fill(this.$options.value.min, previousLength);
      this.$values.set(values);
      this.$options.set(options);
    });

    // console.log(this.$items.value, this.$options.value);

  }

  mount(target?: HTMLElement): void {
    const t = target || document.querySelector(`#${this.id}`);
    if (!t) return;
    this.destroy();
    this.$$.app = new View({
      target: t,
      props: {
        title: this.title,
        items: this.$items,
        values: this.$values,
        options: this.$options,
      },
    });
  }
}
