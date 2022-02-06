import { Timer, TimerOptions } from './timer.component';

export function timer(options?: TimerOptions): Timer {
  return new Timer(options);
}

export type { Timer };
