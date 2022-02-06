import { ClassifierResults, MLPClassifier, mlpClassifier, MLPClassifierOptions, Model } from '@marcellejs/core';
import { TensorLike } from '@tensorflow/tfjs-core';
import { 
  DeepEnsemble, 
  DeepEnsembleOptions,
} from './deep-ensemble.module';

/**
 * 
 * @param modelGeneratorFunc a function that returns a model initialised with random weights.
 * @param options includes the size of the ensemble (defaults to 3).
 * @returns
 */
export function deepEnsemble<SubModel extends Model<InputType, ClassifierResults>, InputType>(
  modelGeneratorFunc: () => SubModel, 
  options?: Partial<DeepEnsembleOptions>): DeepEnsemble<SubModel, InputType>{
  return new DeepEnsemble<SubModel, InputType>(
    modelGeneratorFunc,
    options
  );
}

export function deepEnsembleMlp(
  options?: Partial<DeepEnsembleOptions>, 
  mlpOptions?: Partial<MLPClassifierOptions>) {
  return new DeepEnsemble<MLPClassifier, TensorLike>(
    () => mlpClassifier(mlpOptions),
    options
  );
}

export type { DeepEnsemble, DeepEnsembleOptions };
