<svelte:options accessors />

<script lang="ts">
  import { Stream } from '@marcellejs/core';
  import { onDestroy, onMount } from 'svelte';

  import { KeyboardKallback } from './keyboard-input.component';

  export let keydown: Stream<KeyboardKallback>;
  export let keyup: Stream<KeyboardKallback>;
  // export let options: KeyboardInputOptions;

  onMount(() => {
    window.onkeydown = $keydown;
    window.onkeyup = $keyup;
  });

  onDestroy(() => {
    window.onkeydown = null;
    window.onkeyup = null;
  });
</script>
