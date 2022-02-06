<svelte:options accessors />

<script lang="ts">
  import { ViewContainer, Stream } from '@marcellejs/core';
  import { Switch } from '@marcellejs/core/ui';
  import RangeSlider from 'svelte-range-slider-pips';

  import TextArea from '../../ui/components/TextArea.svelte';
  import { SurveyOptions, SurveyQuestion, SurveyAnswer } from './survey.component';

  export let title: string;
  export let questions: Stream<Array<SurveyQuestion>>;
  export let answers: Stream<Array<SurveyAnswer>>;
  export let options: Stream<SurveyOptions>;

  function likertMax(levels: number | string[]) {
    return (Array.isArray(levels) ? (levels as string[]).length : levels) - 1;
  }
  function likertFormatter(levels: number | string[]) {
    return Array.isArray(levels)
      ? (val: number) => (levels as string[])[val]
      : (val: number) => val;
  }

  // For some reason, binding the 'values' to any kind of stream
  // results in endless loading, (but does not in other SliderArray or SliderChart),
  // may be because of the step? or pips.
  // So we circumvent this by using on:change.
  // see slider events (dispatched) section.
  // https://github.com/simeydotme/svelte-range-slider-pips
  function onChange(idx: number) {
    return (e: any) => {
      $answers[idx].data.level = e.detail.value;
    };
  }
</script>

<ViewContainer {title}>
  <!-- Slider if Likert-->
  <!-- todo: formatter for RangeSlider -->
  <!-- TextField if open-->
  <div class="component container mx-auto h-full">
    <div class="survey">
      {#if $questions && $answers}
        {#each $questions as question, idx}
          <div class="w-full box-content">
            <p class="font-semibold">{@html question.text}</p>
            {#if question.options.explanation}
              <p class="font-italic">{@html question.options.explanation}</p>
            {/if}
            <div class="answer">
              {#if question.options?.skippable}
                <Switch text={question.options.skip.text} bind:checked={$answers[idx].skipped} />
              {/if}
              {#if question.format === 'likert'}
                <div class="likert">
                  <RangeSlider
                    on:change={onChange(idx)}
                    min={0}
                    step={1}
                    float={false}
                    vertical={false}
                    pips={true}
                    pipstep={1}
                    all={'label'}
                    springValues={$options.slider.springValues}
                    max={likertMax(question.options.likert.levels)}
                    formatter={likertFormatter(question.options.likert.levels)}
                  />
                </div>
              {:else if question.format === 'text'}
                <div class="text w-full h-40">
                  <TextArea
                    bind:value={$answers[idx].data.text}
                    placeholder={question.options.text.placeholder}
                    maxLength={question.options.text.max_length}
                  />
                </div>
              {:else if question.format === 'open'}
                <div class="open w-full h-40" />
              {:else if question.format === 'radio'}
                <div class="radio w-full h-40">
                  {#each question.options.radio.values as value}
                    <label>
                      <input
                        type={question.options.radio.type}
                        name={question.options.radio.name}
                        {value}
                      />
                      {value}
                    </label>
                  {/each}
                </div>
              {/if}
            </div>
          </div>
        {/each}
      {/if}
    </div>
  </div>
</ViewContainer>

<style lang="postcss">
  .survey {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: space-between;
  }
</style>
