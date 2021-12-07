import { Stream } from '@marcellejs/core';
import { ButtonSelect, ButtonSelectOptions } from './button-select.component';

export function buttonSelect(items: Stream<string[]>, values: Stream<any[]>, options?: Partial<ButtonSelectOptions>): ButtonSelect {
  return new ButtonSelect(items, values, options);
}

export type { ButtonSelect };
