import { map, startWith } from '@most/core';
import { Component, Stream, Prediction, Text, text } from '@marcellejs/core';

export class ClassificationLabel extends Component {
  title = 'classification label';

  displayLabel: Text;

  constructor(predictionStream: Stream<Prediction>) {
    super();
    this.displayLabel = text({ text: 'Waiting for predictions...' });
    this.displayLabel.title = this.title;
    this.displayLabel.$text = new Stream(
      startWith('Waiting for predictions...')(
        map(({ label, trueLabel }: Prediction) => {
          let t = `<h2>Predicted Label: <code>${label}</code></h2>`;
          if (trueLabel !== undefined) {
            t += `<p>True Label: ${trueLabel} (${
              label === trueLabel ? 'Correct' : 'Incorrect'
            })</p>`;
          }
          return t;
        }, predictionStream),
      ),
      true,
    );
    this.start();
  }

  mount(target?: HTMLElement): void {
    const t = target || document.querySelector(`#${this.id}`);
    if (!t) return;
    const divLab = document.createElement('div');
    divLab.id = `${t.id}-${this.displayLabel.id}`;
    const divConf = document.createElement('div');
    t.appendChild(divLab);
    t.appendChild(divConf);
    this.displayLabel.title = this.title;
    this.displayLabel.mount(divLab);
    this.destroy = () => {
      divLab.parentElement.removeChild(divLab);
      divConf.parentElement.removeChild(divConf);
      this.displayLabel.destroy();
    };
  }

  destroy(): void {
    this.displayLabel.destroy();
  }

}
