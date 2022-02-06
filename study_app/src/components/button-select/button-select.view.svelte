<svelte:options accessors />

<script lang="ts">
  import { ViewContainer, Stream } from '@marcellejs/core';
  import { Button } from '@marcellejs/core/ui';

  export let title: string;

  export let items: Stream<string[]>;
  export let values: Stream<any[]>;
  export let selected: Stream<any | null>; // value of pressed

  export let pressedArr: Stream<boolean[]>;
  export let click: Stream<CustomEvent<unknown>>;
  export let loading: Stream<boolean>;
  export let disabled: Stream<boolean[]>;
  export let type: Stream<'default' | 'success' | 'warning' | 'danger'>;

  function handleClick(idx: number) {
    return (event: CustomEvent) => {
      selected.set($values[idx]);
      click.set(event);
    };
  }
</script>

<ViewContainer {title} loading={$loading}>
  {#if $items && $values}
    <div class="flex-grow">
    {#each $items as item, idx}
      <Button
        disabled={$disabled[idx]}
        type={$type}
        bind:pressed={$pressedArr[idx]}
        on:click={handleClick(idx)}
      >
        {item}
      </Button>
    {/each}
    </div>
  {/if}
</ViewContainer>
