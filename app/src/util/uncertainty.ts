
import { Uncertainty, EnsembleResults, DensityEstimatorResults } from '../types';
import { std } from 'mathjs';
export class UncertaintyUtility {

  public static computeUncertainty(resultEmbeddings: EnsembleResults, resultDensity: DensityEstimatorResults): Uncertainty{
    // const variationRatio = UncertaintyUtility.variationRatio(resultEmbeddings.perModelLabels);
    const entropy = UncertaintyUtility.efficiency(resultEmbeddings.perClassMeanConfidences);
    // const bald = UncertaintyUtility.bald(entropy, resultEmbeddings.perModelConfidences);
    // const meanStandardDeviation = UncertaintyUtility.meanStandardDeviation(resultEmbeddings.perClassConfidences);
    // todo
    const gmmLikelihood = UncertaintyUtility.gmmLikelihood(resultDensity);
    return {
      aleatoric: entropy,
      epistemic: gmmLikelihood,
      // variationRatio,
      // bald,
      // meanStandardDeviation,
    }
  }

  public static computeUncertaintyDiff(ground: Uncertainty, answer: Uncertainty) {
    return Object.fromEntries(
      Object.entries(answer).map(
        ([name, value]): [string, number] => 
          [name, value - ground[name]]
      )
    ) as Uncertainty;
  }

  public static computeUncertaintyDist(ground: Uncertainty, answer: Uncertainty) {
    return Object.fromEntries(
      Object.entries(answer).map(
        ([name, value]): [string, number] => 
          [name, Math.abs(value - ground[name])]
      )
    ) as Uncertainty;
  }

  public static computeUncertaintyScore(ground: Uncertainty, answer: Uncertainty) {
    return Object.fromEntries(
      Object.entries(answer).map(
        ([name, value]): [string, number] => 
          [name, Math.abs(value - Math.max(0, Math.min(1, ground[name])))]
      )
    ) as Uncertainty;
  }

  /**
   * 
   * @param predictions 
   * @returns 
   */
  public static variationRatio(perModelLabels: Array<string>): number {
    // for now, we use the 'Variation Ratio' measure.
    // count occurrences of elements for each class.
    const uniqueLabels = Array.from(new Set(perModelLabels));
    const predictionCounts = uniqueLabels
      .map(l1 => 
        perModelLabels
          .filter(l2 => 
            l1 === l2).length
      );
    //
    return 1 - Math.max(...predictionCounts) / perModelLabels.length;
  }

  /**
   * 
   * @param perClassMeanConfidences 
   * @returns 
   */
  public static margin(perClassMeanConfidences: Record<string, number>) {
    const sortedConfidences = Object.values(perClassMeanConfidences)
      .sort((a ,b) => b - a /* largest first */);
    return sortedConfidences[0] - sortedConfidences[1];
  }

  /**
   * 
   * @param perClassConfidences 
   * @returns 
   */
  public static meanStandardDeviation(perClassConfidences: Record<string, Array<number>>) {
    let meanDeviation = 0;
    const classes = Object.values(perClassConfidences);
    for (const label in perClassConfidences) {
      const confidences = perClassConfidences[label];
      const stdConfidences = std(confidences)
      meanDeviation += stdConfidences
    }
    meanDeviation /= classes.length
    return meanDeviation;
  }

  /**
   * Shannon's entropy as an uncertainty measure.
   * @param perClassMeanConfidences 
   * @returns 
   */
  public static entropy(perClassMeanConfidences: Record<string, number>): number {
    const values  = Object.values(perClassMeanConfidences);
    return values.reduce((acc, p) => acc - (Math.abs(p) > Number.EPSILON * 2 ? p * Math.log(p) : 0.0), 0);
  }

  /**
   * Normalised entropy
   * see: https://en.wikipedia.org/w/index.php?title=Entropy_(information_theory)&section=14#Efficiency_(normalized_entropy)
   * @param perClassMeanConfidences
   * @returns 
   */
  public static efficiency(perClassMeanConfidences: Record<string, number>): number {
    // normalisation.
    const ent = this.entropy(perClassMeanConfidences); 
    const eff =  ent / Math.log(Object.keys(perClassMeanConfidences).length);
    return eff * eff;
  }

  /**
   * Bayesian Learning By Disagreement (BALD) uncertainty measure.
   * @param entropy 
   * @param perModelConfidences 
   * @returns 
   */
  public static bald(
    entropy: number, 
    perModelConfidences: Array<Record<string, number>>
    ) {
    let conditionalEntropy = 0;
    for (let i = 0; i < perModelConfidences.length; ++i) {
      const ps = Object.values(perModelConfidences[i]);
      for (let j = 0; j < ps.length; ++j) {
        conditionalEntropy -= ps[j] * Math.log(ps[j]);
      }
    }
    conditionalEntropy /= perModelConfidences.length;
    return entropy - conditionalEntropy;
  }

  /**
  * 
  * @param likelihood
  * @returns 
  */
  public static gmmLikelihood(resultDensity: DensityEstimatorResults) {
    // for now, we use the 'Variation Ratio' measure.
    // count occurrences of every element
    // log(p) --> np.exp(p) in [0, 1]
    // const likelihood = 1 - Math.exp(logLikelihood);
    // METHOD1: Normalise log probability density
    /*
    const logDensity = resultDensity.logDensity;
    const logMin = resultDensity.bounds.min;
    const logMax = resultDensity.bounds.max;
    // fixme: should we handle division by zero? nah.
    const norm = (v: number, min: number, max: number) => (v - min) / (max - min);
    const novelty = 1 - norm(logDensity, logMin, logMax);
    */
    // METHOD2: Normalise pseudo-probability
    // --> can't work because exp(...) is too close to zero.
    /*
    const p = (v: number) => Math.exp(v);
    const novelty2 = 1 - norm(p(logLikelihood), p(logMin), p(logMax));
    */
    const novelty = resultDensity.distance;
    return novelty
  }
}