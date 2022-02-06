import { Component, Stream } from '@marcellejs/core';
import {Uncertainty} from '../../types';
import { genericChart, GenericChart } from '@marcellejs/core';
import { map } from '@most/core';




export class UncertaintyPlot extends Component {
  title = 'uncertainty-plot [custom module 🤖]';

  $valueStream: Stream<{x: string; y: number}[]>;
  chart: GenericChart;


  constructor(uncertaintyStream: Stream<Uncertainty>) {
    super();
    this.$valueStream = new Stream(
      map( (uncertainty) => 
      Object.entries(uncertainty).map(([label, value]) => ({ x: label, y: value })),
        uncertaintyStream,
      ),
    );


    this.chart = genericChart({
      preset: 'bar-fast',
      options: {
        aspectRatio: 2,
        xlabel: 'Type of uncertainty',
        ylabel: '%',
        scales: { y: { suggestedMax: 1 } },
      },
    });
    this.chart.addSeries(
      this.$valueStream as Stream<number[]> | Stream<Array<{ x: unknown; y: unknown }>>,
        'Uncertainties',
    );
    this.chart.title = '';
    this.start();

  }

  mount(target?: HTMLElement): void {
    const t = target || document.querySelector(`#${this.id}`);
    if (!t) return;
    const divLab = document.createElement('div');
    // divLab.id = `${t.id}-${this.#displayLabel.id}`;
    const divConf = document.createElement('div');
    divConf.id = `${t.id}-${this.chart.id}`;
    t.appendChild(divLab);
    t.appendChild(divConf);
    // this.#displayLabel.title = this.title;
    // this.#displayLabel.mount(divLab);
    this.chart.mount(divConf);


    // todo: why are there two destroy methods?
    this.destroy = () => {
      divLab.parentElement.removeChild(divLab);
      divConf.parentElement.removeChild(divConf);
      // this.#displayLabel.destroy();
      this.chart.destroy();
    };
  }
  destroy(): void {
    this.chart.destroy();
  }
}
