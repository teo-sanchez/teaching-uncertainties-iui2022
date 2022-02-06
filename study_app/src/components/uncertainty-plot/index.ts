import {Uncertainty} from '../../types';
import {Stream} from '@marcellejs/core'
import {UncertaintyPlot} from './uncertainty-plot.module';

export function uncertaintyPlot(uncertaintyStream: Stream<Uncertainty>): UncertaintyPlot {
  return new UncertaintyPlot(uncertaintyStream);
}

export type { UncertaintyPlot };
