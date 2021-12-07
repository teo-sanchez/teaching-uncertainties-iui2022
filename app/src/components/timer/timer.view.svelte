<svelte:options accessors />

<script lang='ts'>
  import { ViewContainer, Stream } from '@marcellejs/core';
  import { TimerTime } from './timer.component';

  export let title: string;
  export let time: Stream<TimerTime>;

  function getTimeString(el: number) {
    //see comment by Henrik N: https://stackoverflow.com/a/12612778
    return new Date(null, null, null, null, null, null, el).toTimeString().split(' ')[0];
  }
</script>

<ViewContainer {title}>
  <div class="flex-grow flex">
    {#if $time.state !== 'stopped'}
      <p class="font-medium text-2xl">
      {#if $time.countdown}
        {@html getTimeString($time.elapsed) + '/' + getTimeString($time.countdown)}
      {:else}
        {@html getTimeString($time.elapsed)}
      {/if}
      </p>
    {/if}
  </div>
</ViewContainer>

<style>
</style>
