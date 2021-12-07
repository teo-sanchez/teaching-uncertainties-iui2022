<script>
  import { ModuleBase } from '@marcellejs/core';

  export let title;
  export let options;
</script>

<style>
  .my-color {
    color: seagreen;
  }
</style>

<svelte:options accessors />

<ModuleBase {title}>
  <div>This is a <span class="my-color">custom</span> module with the following options:</div>
  <p>{JSON.stringify(options)}</p>
</ModuleBase>
