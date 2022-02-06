<svelte:options accessors />

<script lang="ts">
  import { ViewContainer, Stream } from '@marcellejs/core';
  import RangeSlider from 'svelte-range-slider-pips';

  import { VerticalSliderOptions } from '../../types';

  export let title: string;
  export let items: Stream<string[]>;
  export let values: Stream<number[]>;
  export let options: Stream<VerticalSliderOptions[]>;

  // i -> n - i [0, n]
  // i -> n - i + 1 [0, n -1]
  // For some reason, binding the 'values' to any kind of stream
  // results in endless loading, (but does not in other SliderArray or SliderChart),
  // may be because of the step? or pips.
  // So we circumvent this by using on:change.
  // see slider events (dispatched) section.
  // https://github.com/simeydotme/svelte-range-slider-pips
  function onChange(idx: number) {
    const opts = $options[idx];
    return (e: any) => {
      values.set([
        ...$values.slice(0, idx),
        opts.max + opts.min - e.detail.value, // values are flipped!
        ...$values.slice(idx + 1),
      ]);
    };
  }

  $: sliderValues = $values.map((val, idx) => [$options[idx].min + $options[idx].max - val]);

  $: colourStyle = $options.map(
    (opt) =>
      `--range-handle: ${opt.colour?.main};
    --range-handle-inactive: ${opt.colour?.main};
    --range-range-inactive: ${opt.colour?.main};
    --range-float: ${opt.colour?.main};
    --range-range: ${opt.colour?.focussed};
    --range-handle-focus: ${opt.colour?.focussed};
    `,
  );

  function formatter(options: VerticalSliderOptions) {
    // IMPORTANT:
    // values are flipped in the slider!
    return (value: any) => options.formatter(options.max + options.min - value);
  }
</script>

<ViewContainer {title}>
  <div class="container">
    <div class="component">
      {#if $items && $values}
        {#each $items as it, idx}
          <!-- bind values-->
          <div class="slider" style={colourStyle[idx]}>
            <RangeSlider
              on:change={onChange(idx)}
              bind:value={sliderValues[idx]}
              min={$options[idx].min}
              max={$options[idx].max}
              range={'max'}
              float={$options[idx].float}
              vertical={true}
              step={$options[idx].step}
              pipstep={$options[idx].pipstep}
              pips={$options[idx].pips}
              all={$options[idx].all}
              rest={$options[idx].rest}
              first={$options[idx].first}
              last={$options[idx].last}
              springValues={$options[idx].springValues}
              formatter={formatter($options[idx])}
            />
            <div class="slider-title">
              <p class="text-semibold">{@html it}</p>
            </div>
          </div>
        {/each}
      {/if}
    </div>
  </div>
</ViewContainer>

<style lang="postcss">
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
