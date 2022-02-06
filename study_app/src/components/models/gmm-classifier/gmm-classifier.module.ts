import { 
  Dataset, 
  ObjectId, 
  Parametrable,
  StoredModel,
  Model,
  ModelOptions,
} from '@marcellejs/core';

import { Utility } from '../../../util';
import { DensityEstimatorResults } from '../../../types';

import { Matrix, covariance } from 'ml-matrix';
import { GMM } from '../../../deps/gaussian-mixture-model';

export interface GMMClassifierOptions extends Partial<ModelOptions> {
  nOptimization: number;
  bufferSize: number;
}

/**
 * Gaussian Mixture Model (GMM)
 * Modified from: https://github.com/lukapopijac/gaussian-mixture-model
 */
export class GMMClassifier extends Model<number[][], DensityEstimatorResults> {

  title = 'Gaussian Mixture Model';
  options: GMMClassifierOptions;
  model: GMM;

  private weights: number[];
  private means: number[][];
  private covariances: number[][][];

  constructor(options?: Partial<GMMClassifierOptions>) {
    super(options);
    this.options = {
      nOptimization: 1,
      bufferSize: 1e6,
      ...options,
    };

  }

  parameters: Parametrable['parameters'];
  serviceName: 'gaussian-mixture-model';

  private build(xs: number[][], ys: string[]): void {
    this.classes = Array.from(new Set(ys));
    this.numberFeatures = xs[0].length;

    // create a dictionary indexing the features by class.
    const xc: Record<string, Matrix> = {};
    for (const c of this.classes) {
      // indices of the instances belonging to this class.
      const ids = ys.filter(y => y === c).map((_, idx) => idx);
      // features of the elements belonging to this class.
      // todo: optimise.
      xc[c] = new Matrix(xs.filter((_, idx) => ids.some(x => x === idx)));
    }
    // PRIORS
    // means
    // todo: not sure about 'column', may be 'row'.
    this.means = Object.values(xc).map(m => m.mean('column'));
    // covs
    this.covariances = Object.values(xc).map(m => covariance(m, { center: true }).to2DArray());
    /*
    this.covariances = new Array(this.classes.length);
    for (let i = 0; i < this.classes.length; ++i) {
      for (let j = i; j < this.classes.length; ++j) {
        const cov = covariance(xc[this.classes[i]], xc[this.classes[j]]);
        const arr = cov.to2DArray();
        this.covariances[i] = arr;
        this.covariances[j] = arr;
      }
    }
    */
    // weights
    this.weights = new Array(this.classes.length).fill(1 / this.classes.length);
    //
    this.model = new GMM({
      weights: this.weights,
      means: this.means,
      covariances: this.covariances,
      bufferSize: this.options.bufferSize
    });
    // Adding all points
    xs.forEach(x => this.model.addPoint(x));
    //
  }

  // @Catch
  async train(dataset: Dataset<number[][], string>): Promise<void> {
    //First need to initialize means and covariance around the data
    //what for? --> displaying the state of the model in the 'training progress' component.
    this.$training.set({ status: 'loading' }); 
    const instances = await dataset.items().toArray();
    
    const xs = instances.map(inst => inst.x.flat());
    const ys = instances.map(inst => inst.y);

    this.build(xs, ys);

    this.$training.set({ 
      status: 'start',
      epochs: this.options.nOptimization
    }); 
    // Perform Expectatio-Maximization
    this.model.runEM(
      this.options.nOptimization,
      (it: number) => {
        this.$training.set({
          status: 'epoch',
          epochs: this.options.nOptimization,
          epoch: it,
        })
    });
    this.$training.set({ status: 'success'});
  }

  async predict(features: number[][]): Promise<DensityEstimatorResults> {
    // todo
    if (!this.model) { 
      return null;
    }
    const x = features.flat();
    if (x.length != this.numberFeatures){
      throw new RangeError(`Number of features inconsistent: found ${x.length}, expected ${this.numberFeatures}`);
    }
    const probabilities = await this.model.predictNormalize(x) as number[];
    const label = this.classes[Utility.argMaxArr(probabilities)];
    
    return {
      label: label, 
      confidences: Object.fromEntries(probabilities.map((p, idx) => [this.classes[idx], p])),
      distance: 0, // todo
      bounds: { // todo
        min: 0,
        max: 0,
      } 
    };
  }

  // @checkProperty('dataStore')
  async save(
    name: string,
    metadata?: Record<string, unknown>, 
    id: ObjectId = null,
  ): Promise<ObjectId | null> {
    // todo
    return null; 
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

  private classes: Array<string>;
  private numberFeatures: number;

}
