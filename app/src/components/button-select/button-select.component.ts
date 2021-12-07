import { never } from '@most/core';
import { Component, Stream } from '@marcellejs/core';
import View from './button-select.view.svelte';

export interface ButtonSelectOptions {
  type?: 'default' | 'success' | 'warning' | 'danger';
}

export class ButtonSelect extends Component {
  title = 'button-select [custom module 🤖]';

  $items: Stream<string[]>;
  $values: Stream<any[]>;

  // does not work, see how to induce $stream.value to derived streams.
  // here: https://github.com/mostjs/adapter
  $pressed: Stream<boolean>; // if one of the buttons is pressed.
  // $click: Stream<CustomEvent<unknown>>;
  $click = new Stream<CustomEvent<unknown>>(never());
  $loading = new Stream(false, true);
  $disabled: Stream<Array<boolean>>;
  $type: Stream<'default' | 'success' | 'warning' | 'danger'>;

  $selected: Stream<any | null>;
  private $pressedArr: Stream<Array<boolean>>;

  constructor(
    items: Stream<string[]>,
    values: Stream<any[]>,
    options?: Partial<ButtonSelectOptions>) {
    super();
    options = {
      type: 'default',
      ...options
    };
    this.$items = new Stream(items, true);
    this.$items.start();
    this.$type = new Stream(options.type, true);
    this.$selected = new Stream(null, true);
    // $value is iderived from $selected and $items.
    // it is the value of the selected item.
    this.$values = new Stream(values, true);
    this.$values.start();
    this.$pressedArr = new Stream([], true);
    this.$pressedArr.start();
    // resize $disabled and $pressedArr depending on the number of items.
    this.$disabled = new Stream([], true);
    this.$disabled.start();
    this.$items.subscribe(its => {
      const disabled = this.$disabled.value;
      const pressedArr = this.$pressedArr.value;
      const previousLength = disabled.length;
      disabled.length = its.length;
      pressedArr.length = its.length;
      //
      disabled.fill(false, previousLength);
      pressedArr.fill(false, previousLength);
      this.$disabled.set(disabled);
      this.$pressedArr.set(pressedArr);
    });
    this.$pressed = this.$pressedArr.map(
      pressedArr => pressedArr.some(pressed => pressed)
    );
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
        pressedArr: this.$pressedArr,
        selected: this.$selected,
        click: this.$click,
        loading: this.$loading,
        disabled: this.$disabled,
        type: this.$type
      },
    });
  }
}
