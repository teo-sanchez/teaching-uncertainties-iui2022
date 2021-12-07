<svelte:options accessors />

<script lang="ts">
  import { ViewContainer, Stream } from '@marcellejs/core';
  import RangeSlider from 'svelte-range-slider-pips';

  import { VerticalSliderOptions } from '../../types';
  import { SliderChartData } from './slider-chart.component';

  export let title: string;
  export let data: Stream<Array<SliderChartData>>;
  export let options: Stream<VerticalSliderOptions>;

  function formatter(options: VerticalSliderOptions) {
    // IMPORTANT:
    // values are flipped in the slider!
    return (value: any) => options.formatter(options.max + options.min - value);
  }
</script>

<ViewContainer {title}>
  <div class="container">
    <div class="component">
      {#if $data && $options}
        {#each $data as d}
          <div
            class="slider"
            style="--range-handle:{d.colour?.main};
      --range-handle-inactive: {d.colour?.main};
      --range-range-inactive: {d.colour?.main};
      --range-range: {d.colour?.focussed};
      --range-handle-focus: {d.colour?.focussed}
        "
          >
            <RangeSlider
              values={[$options.max - d.y]}
              min={$options.min}
              max={$options.max}
              range={'max'}
              float={$options.float}
              vertical={true}
              step={$options.step}
              pips={$options.pips}
              pipstep={$options.pipstep}
              all={$options.all}
              rest={$options.rest}
              first={$options.first}
              last={$options.last}
              springValues={$options.springValues}
              formatter={formatter($options)}
            />
            <div class="slider-title">
              <p class="text-semibold">{@html d.x}</p>
            </div>
          </div>
        {/each}
      {/if}
    </div>
  </div>
</ViewContainer>

<style lang="css">
  .component {
    display: flex;
    flex-direction: row;
    justify-content: space-around;
  }

  .slider {
    display: flex;
    flex-direction: column;
    align-items: center;
  }
</style>
