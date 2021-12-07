import { GMMClassifier, GMMClassifierOptions } from './gmm-classifier.module';

export function gmmClassifier(options?: Partial<GMMClassifierOptions>): GMMClassifier {
  return new GMMClassifier(options);
}

export type { GMMClassifier };
