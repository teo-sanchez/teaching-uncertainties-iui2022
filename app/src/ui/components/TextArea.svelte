<script lang="ts">
  export let placeholder = '';
  export let disabled = false;
  // export let type: 'text' | 'email' | 'passowrd' = 'text';
  export let value: string = null;
  export let maxLength: number = null;
  // export let resizable = false;

  function handleInput(e: Event & { currentTarget: EventTarget & HTMLInputElement }) {
    value = e.currentTarget.value;
  }
</script>

<!-- todo: on:input and resizable -->
<textarea class="marcelle" {disabled} {maxLength} {placeholder} {value} on:input={handleInput} />

<style lang="postcss">
  .marcelle {
    @apply w-full px-4 py-2 rounded mr-1 mb-1 bg-white text-gray-600 border border-solid border-gray-300;
    width: 100%;
    height: 80%;
    transition: all 0.15s ease;
    resize: none;
  }

  .marcelle::placeholder {
    @apply text-gray-300;
  }

  .marcelle:hover {
    @apply border-gray-400;
  }

  .marcelle:focus {
    @apply outline-none ring-blue-400 ring-2 ring-opacity-50;
  }

  .marcelle:active {
    @apply ring-blue-400 ring-4 ring-opacity-50;
  }

  .marcelle:invalid {
    @apply border-red-500 text-red-600;
  }

  .marcelle:invalid:focus {
    @apply ring-red-300;
  }
</style>
