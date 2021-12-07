import type { Stream, Prediction } from '@marcellejs/core';
import { ClassificationLabel } from './classification-label.module';

export function classificationLabel(predictionStream: Stream<Prediction>): ClassificationLabel {
  return new ClassificationLabel(predictionStream);
}

export type { ClassificationLabel };
