import { KeyboardInput, KeyboardKallback, KeyboardInputOptions } from './keyboard-input.component';

export function keyboardInput(options?: Partial<KeyboardInputOptions>): KeyboardInput {
  return new KeyboardInput(options);
}

export type { KeyboardInput };
export { KeyboardKallback, KeyboardInputOptions };
