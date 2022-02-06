import { Component, Stream } from '@marcellejs/core';
import View from './timer.view.svelte';

/**
 * accuracy Number of decimals after second to include.
 */
export interface TimerOptions {
  accuracy: number;
}

export interface TimerTime {
  start: Date, 
  last: Date,
  stop: null | Date,
  elapsed: number, // seconds
  countdown: null | number, // seconds
  state: 'started' | 'stopped' | 'paused' | 'running',
}

/**
 * Accurate timer;
 * see: https://stackoverflow.com/a/29972322
 */
export class Timer extends Component {
  title = 'timer [custom module 🤖]';

  options: TimerOptions;
  $time: Stream<TimerTime>;

  constructor(options: TimerOptions = {
    accuracy: 0,
  }) {
    super();
    this.options = options;
    this.$time = new Stream({
      start: null,
      last: null,
      state: 'stopped',
      stop: null,
      countdown: null,
      elapsed: 0,
    } as TimerTime, true);
  }


  /**
   * 
   * @param countdown if present, then stopped after countdown milliseconds.
   * @param interval timestep for checking, in milliseconds.
   */
  public startTimer(opt: {
    countdown?: number,
    interval?: number
  }) {
    if (!opt.interval) {
      opt.interval = 1000 / (2 * (this.options.accuracy + 1));
    }
    this.interval = opt.interval;
    const begin = new Date();
    this.$time.set({
      ...this.$time.value,
      start: begin,
      elapsed: 0,
      stop: null,
      countdown: opt.countdown,
      state: 'started'
    });
    this.launch(this.interval);
  }

  /**
   * 
   * @param shouldPause if true, pause, else resume.
   */
  public pauseTimer(shouldPause: boolean) {
    const isPaused = this.$time.value.state === 'paused';
    this.$time.set({
      ...this.$time.value,
      state: shouldPause ? 'paused' : 'started'
    });
    if (shouldPause && !isPaused) {
      this.clear();
    }
    else if (!shouldPause && isPaused) {
      this.launch(this.interval);
    }
  }

  public stopTimer(): void {
    // todo
    const isStopped = this.$time.value.state === 'stopped';
    if (isStopped) {
      return;
    }
    this.$time.set({
      ...this.$time.value,
      stop: new Date(),
      state: 'stopped'
    });
    this.clear();
  }

  mount(target?: HTMLElement): void {
    const t = target || document.querySelector(`#${this.id}`);
    if (!t) return;
    this.destroy();
    this.start();
    this.$$.app = new View({
      target: t,
      props: {
        title: this.title,
        time: this.$time,
      },
    });
  }

  private launch(expected: number) {
    const step = () => {
      // from: https://stackoverflow.com/a/29972322
      const time = this.$time.value;
      const current = new Date();
      const drift = (current.getTime() - time.last.getTime()) - expected;
      if (Math.abs(drift) > this.interval) {
        // todo: big lag, throw exception or something.
      }
      this.$time.set({
        ...time,
        last: current,
        elapsed: time.elapsed + expected + drift,
        state: 'running',
      });
      if (time.countdown && time.elapsed >= time.countdown) {
          this.stop();
      }
      else {
        this.timeout = setTimeout(step, Math.max(0, this.interval - drift));
      }
    };
    this.$time.set({
      ...this.$time.value,
      last: new Date(),
    });
    this.timeout = setTimeout(step, this.interval);
  }

  private clear() {
    if (this.timeout) {
      clearTimeout(this.timeout);
    }
  }

  private interval: number; // milliseconds;
  private timeout: any; // actually number, or NodeJS.Timeout, according to env...
}
