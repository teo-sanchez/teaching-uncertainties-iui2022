import { 
  ClassifierResults, 
  Dataset, 
  ObjectId, 
  Parametrable,
  StoredModel,
  Model,
  ModelOptions
} from '@marcellejs/core';

import { Matrix, covariance } from 'ml-matrix';

import { Utility } from '../../../util';
import { multivariateNormal } from '../../../util';



/**
 * Gaussian Discriminant Analysis (GDA) Model
 * todo: Figure out the right input type.
 * For reference, see:
 * J. Mukhoti, A. Kirsch, J. van Amersfoort, P. H. S. Torr, and Y. Gal, 
 * “Deterministic Neural Networks with Appropriate Inductive Biases Capture Epistemic and Aleatoric Uncertainty,” 2021, 
 * [Online].  * Available: http://arxiv.org/abs/2102.11582.
 */
export class GDA extends Model<number[][], ClassifierResults> {
  title = 'Gaussian Discriminent Analysis (GDA)';

  constructor(options?: Partial<ModelOptions>) {
    super(options);
  }

  parameters: Parametrable['parameters'];
  serviceName: 'gda-model';

  /**
   * todo: why isn't train it async in Model? 
   * They still use asynchronous code in MLP with setTimeout, 
   * why not put it in the interface?
   * @param dataset The training data set.
   */
  // @Catch
  async train(dataset: Dataset<number[][], string>): Promise<void> {
    // translated to js from python in: uncertainty_benchmark/models.py
    // todo: add more info in training stream.
    this.$training.set({ status: 'start' });
    //
    const instances = await dataset.items().toArray();
    const numberInstances = instances.length;
    // xs are flattened features.
    // todo: not sure about that actually,
    // features is a number[][] in Instance,
    // and from the MLP model source file, it seems that the first dimension of the features
    // is the index of the example, but we should only ever get one,
    // since we only have one label per instance..
    const xs = instances.map(inst => inst.x.flat());
    const ys = instances.map(inst => inst.y);
    //
    this.classes = Array.from(new Set(ys));
    this.numberFeatures = xs[1].length;
    //
    console.log('Number of classes:  ', this.classes.length);
    console.log('Number of features: ', this.numberFeatures);
    // Initialisation
    this.mu = {};
    this.sigma = {};
    this.phi = {};
    //
    for (const it in this.classes) {
      const label = this.classes[it];
      // indices of the instances belonging to this class.
      const classIndices = ys.filter(y => y === label).map((_, idx) => idx);
      // features of the elements belonging to this class.
      // todo: optimise.
      const classXs = xs.filter((_, idx) => classIndices.some(el => el === idx));
      // Class init
      this.mu[label] = new Array(this.numberFeatures);
      this.sigma[label] = new Matrix(this.numberFeatures, this.numberFeatures);

      // computing average features for each examples of this class.
      this.mu[label] = classXs.reduce((acc, x) => 
        acc.map((el1, idx) => 
          (el1 + x[idx]) 
        ) // computing element-wise addition.
      );
      this.mu[label].forEach(el => el / classIndices.length); // averaging.
      // Our prior is a uniform distribution of the class accross all training examples.
      this.phi[label] = classIndices.length / numberInstances;
      // nee
      this.sigma[label] = covariance(classXs);
    }
    this.sigmaScaled = GDA.computeNormalisedSigma(this.sigma);
    //
    this.$training.set({ status: 'success' });
  }

  async predict(features: number[][]): Promise<ClassifierResults> {
    // todo
    const x = features.flat();
    // some preliminary checks
    if (x.length != this.numberFeatures) {
      // todo: probability not the right error.
      throw new RangeError(`Number of features inconsistent: found ${x.length}, expected ${this.numberFeatures}`);
    }
    //
    let confidences: { [key: string]: number } = { };
      // Normalise the covariance matrices.
    let density = 0;
    let classConditionalConfidences: { [key: string]: number } = { };
    for (const it in this.classes) {
      const label = this.classes[it];
      const dist  = multivariateNormal(this.mu[label], this.sigmaScaled[label].to2DArray());
      classConditionalConfidences[label]=  dist.samplePDF(x);
      density += classConditionalConfidences[label];
    }
    for (const label in this.classes) {
      // Applying Bayes' formula here,
      // with phi[label] as the prior and density as the sum.
      confidences[label] = (this.phi[label] * classConditionalConfidences[label]) / density;
      // confidences[label] = Math.log(this.phi[label]) + dist.sample();
    }
    const label = Utility.argMaxObj(confidences);
    const result = { label, confidences };
    console.log("GDA conf: ", confidences);
    return result;
  }

  // @checkProperty('dataStore')
  async save(
    name?: string,
    metadata?: Record<string, unknown>, 
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


  private static computeNormalisedSigma(sigma: { [key: string]: Matrix }): { [key: string]: Matrix } {
    // Normalise the covariance matrices.
    const sigmaArr = Object.values(sigma);
    const meanCov = sigmaArr.reduce((acc, el) => Matrix.add(acc, el)).divide(sigmaArr.length);
    sigmaArr.forEach(el => el.sub(meanCov));
    sigmaArr.sort((a, b) => b.max() - a.max());
    const covMaxVal = sigmaArr[0].max();
    return Object.fromEntries(
      Object.entries(sigma)
        .map(([label, cov]) => 
          [label, Matrix.div(Matrix.sub(cov, meanCov), covMaxVal)]
        )
    );
  }

  private classes: Array<string>;
  private numberFeatures: number;
  /**
   * Mean of the input features across all training examples for each class.
   */
  private mu: { [key: string]: Array<number> };
  /**
   * A list of the covariance matrices for each class
   * Each covariance (square, symetric) matrix matrix indicates how features vary with one another.
   */
  private sigma: { [key: string]: Matrix };
  /**
   * Prior probability for each class.
   * An expected likelihood for classes before seeing the input data.
   */
  private phi: { [key: string]: number }; 

  // cached
  private sigmaScaled: { [key: string]: Matrix };

}
