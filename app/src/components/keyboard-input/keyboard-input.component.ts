import { Component, Stream } from '@marcellejs/core';
import View from './keyboard-input.view.svelte';

export class KeyboardInputOptions {

}

export type KeyboardKallback = (e: KeyboardEvent) => (Promise<void> | void);

export class KeyboardInput extends Component {
  title = 'keyboard-input [custom module 🤖]';

  $keyup: Stream<KeyboardKallback>;
  $keydown: Stream<KeyboardKallback>;
  $options: Stream<KeyboardInputOptions>;

  $pressed: Stream<boolean>; // if one of the buttons is pressed.

  constructor(options?: Partial<KeyboardInputOptions>) {
    super();

    this.$keyup = new Stream(e => { }, true);
    this.$keydown = new Stream(e => { }, true);
    this.$options = new Stream({
      ...options
    }, true);
  }

  mount(target?: HTMLElement): void {
    const t = target || document.querySelector(`#${this.id}`);
    if (!t) return;
    this.destroy();
    this.$$.app = new View({
      target: t,
      props: {
        keyup: this.$keyup,
        keydown: this.$keydown,
        options: this.$options
      },
    });
  }
}
