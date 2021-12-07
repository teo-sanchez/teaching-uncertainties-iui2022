import { 
  Stream, 
  ClassifierResults, 
  Dataset, 
  ObjectId, 
  Parametrable,
  StoredModel,
  Model,
  ModelOptions,
  isDataset,
  MLPClassifier,
  Instance,
  getTensorFlowTrainingData,
} from '@marcellejs/core';
import { ServiceIterable } from '@marcellejs/core/types/core/data-store/service-iterable';
import { TensorLike } from '@tensorflow/tfjs';


import { Utility } from '../../../util';
import { EnsembleResults } from '../../../types';

export interface DeepEnsembleOptions extends Partial<ModelOptions> {
  numberModels: number;
}

/**
 * Uses an aggregate of a classifier Model class
 * in order to compute uncertainty of prediction
 * over a given data point.
 * todo: use Snapshot ensembles method for optimisation.
 * note: we could later allow different types of 
 * models at the same time (e.g. MLP and CNN).
 * for details, see: Beluch, William H., et al. “The power of ensembles for active learning in image classification.” 
 * Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 
 * 2018.
 * todo: Actual model used as template parameter instead of InputType!
 */
export class DeepEnsemble<SubModel extends Model<InputType, ClassifierResults>, InputType> extends Model<InputType, EnsembleResults> {
  title = 'Deep Ensemble';

  options: Partial<DeepEnsembleOptions>;

  models: Array<SubModel>;

  constructor(
    modelGeneratorFunc: () => SubModel, 
    options: Partial<DeepEnsembleOptions>
    ) {
    super(options);
    this.options = {
      numberModels: 3,
      ...options,
    }
    if (this.options.numberModels <= 0) {
      throw new RangeError("The number of models in the Deep Ensemble should be positive.");
    }

    this.parameters = {
      $numberModels: new Stream(options.numberModels, true)
    };

    this.modelGeneratorFunc = modelGeneratorFunc;
    
  }

  build(numberModels: number) {
    // delegating Model-specific init with the ensemble.
    // Populate the models array with an ensemble of the same kind of classifier,
    // initialised with the generator function.
    this.models = new Array<SubModel>(numberModels);
    for(let i = 0; i < numberModels; ++i) {
      this.models[i] = this.modelGeneratorFunc();
    }
    this.parameters.$numberModels.set(this.models.length);
    // subscribe to each of the model's training status
    // in order to update the state of the ensemble as a whole.
    this.models.forEach((model, idx) => {
      model.$training.subscribe(e => {
        const ensembleTraining = this.$training.value;
        switch(e.status) {
          case 'loading':
            this.$training.set({ ...ensembleTraining, status: 'loading' });
            break;
          case 'start':
            this.$training.set({ ...ensembleTraining, status: 'start' });
            break;
          case 'epoch':
            this.$training.set({ 
              ...ensembleTraining, 
              status:'epoch', 
              epoch: ++ensembleTraining.epoch,
              data: idx == 0 ? e.data : {} // todo: set data according to model #nb!!
            });
            break;
          case 'success':
            if (this.models.every(other => other.$training.value.status === 'success')) {
              this.hasBeenTrained = true;
              this.isTraining = false;
              this.$training.set({ ...ensembleTraining, status: 'success' });
            }
            break;
          case 'idle':
            if (this.models.every(other => other.$training.value.status === 'idle')) {
              this.$training.set({ ...ensembleTraining, status: 'idle' });
            }
          case 'error':
            // quick fix: ignore error of the ensemble.
            this.$training.set({ ...ensembleTraining, status: 'error' });
            break;
          default:
            break;
        }
      });
    });
  }

  parameters: Parametrable['parameters'];
  serviceName: 'deep-ensemble-model';

  // @Catch
  async train(dataset: Dataset<InputType, string>): Promise<void> {
    // todo: add more info in training stream.
    if (this.isTraining) {
      return;
    }
    this.isTraining = true;
    if (!this.models) {
      this.build(this.options.numberModels);
    }
    // The 'number of epochs' of the training of this ensemble is the sum for all models.
    const sumEpochs = this.models.reduce((acc, model) => acc + model.parameters.epochs.value, 0);
    // All models should be trained on the same data!
    this.$training.set({ 
      status: 'loading', 
      epochs: sumEpochs, 
      epoch: 0 
    });
    // get some common data, so as not to fetch it in each model.
    // will only work with MLPs
    const models = this.models as unknown as MLPClassifier[];
    const ds = <ServiceIterable<Instance<TensorLike, string>>><unknown>(isDataset(dataset) ? dataset.items() : dataset);
    const labels = Array.from(new Set(await ds.map(({ y }) => y).toArray()));
    const data = await getTensorFlowTrainingData(ds, labels);
    const promises = models.map(el => el.trainData(data, labels));
    await Promise.all(promises);
  }

  async predict(x: InputType): Promise<EnsembleResults> {
    if (!this.hasBeenTrained) {
      return null;
    }
    // getting the promises as an array of promises.
    // see: https://stackoverflow.com/a/34387006
    const promises = this.models.map(el => el.predict(x));
    
    // wait for all predictions to be made.
    const predictions = await Promise.all(promises) as Array<ClassifierResults>;

    // initialise class labels for confidence dict.
    const perModelConfidences = predictions.map(pred => pred.confidences);
    let perClassConfidences: Record<string, Array<number>> = { };
    {
      const categories = Object.keys(predictions[0].confidences);
      categories.forEach(cat => perClassConfidences[cat] = new Array<number>(this.models.length));
      for (let i = 0; i < predictions.length; ++i) {
        for (const [cat, p] of Object.entries(predictions[i].confidences)) {
          perClassConfidences[cat][i] = p;
        }
      }
    }
    let perClassMeanConfidences: Record<string, number> = {};
    for (const [label, ps] of Object.entries(perClassConfidences)) {
      perClassMeanConfidences[label] = ps.reduce((acc, p) => acc + p, 0) / ps.length;
    }
    const perModelLabels = perModelConfidences.map(conf => Utility.argMaxObj(conf));

    // the final label is the label with max mean confidence
    const label = Utility.argMaxObj(perClassMeanConfidences);
    // todo: compute mean label and per-category mean confidences.
    // and wrap all of this in an UncertaintyResults
    const result = {
      label,
      confidences: perClassMeanConfidences,
      perClassMeanConfidences,
      perClassConfidences,
      perModelConfidences,
      perModelLabels
    };
    return result;
  }

  // @checkProperty('dataStore')
  async save(
    name: string,
    metadata?: Record<string, unknown>, 
    id: ObjectId = null,
  ): Promise<ObjectId | null> {
    // todo
    if (!this.models) {
      return null;
    }
    const promises = new Array<Promise<ObjectId | null>>(this.models.length);
    for (let idx = 0; idx < this.models.length; ++idx) {
      const model = this.models[idx];
      promises[idx] = model.save(`${name}_${idx}`, metadata);
    }
    const objectIds = await Promise.all(promises) as Array<ObjectId | null>;
    const newId = ''.concat(...objectIds);
    return newId;
  }

  load(idOrName: ObjectId | string): Promise<StoredModel> {
    // todo
    return null; 
  }

  download(metadata?: Record<string, unknown>): Promise<void> {
    // todo
    return null; 
  }

  upload(...files: File[]): Promise<StoredModel> {
    // todo
    return null; 
  }

  private modelGeneratorFunc: () => SubModel;
  private hasBeenTrained: boolean;
  private isTraining: boolean;


}
