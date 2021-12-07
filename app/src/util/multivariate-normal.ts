
import { Matrix, determinant, inverse } from 'ml-matrix';

/**
 * Multivariate normal distribution.
 * see: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
 */
export class MultivariateNormal {
    constructor(mu: Array<number>, sigma: Array<Array<number>>) {
        this.setMean(Matrix.columnVector(mu));
        this.setCovariance(new Matrix(sigma));
    }

    setMean(mu: Matrix): void {
        this.mu = mu;
        this.updatePDF();
    }

    /**
     * 
     * @param cov A positive-definite Covariance matrix.
     */
    setCovariance(sigma: Matrix): void {
        this.detSigma = determinant(sigma);
        if (this.detSigma < 0) {
            // not sufficient for the determinant to be positive
            // for cov to be positive-definite,
            // but should check regardless.
            throw new RangeError("sigma should be positive-definite.");

        }
        this.invSigma = inverse(sigma);
        this.updatePDF();
    }

    samplePDF(x: Array<number>): number {
        if (x.length != this.mu.rows) {
            throw new RangeError("x should be of the same dimension as mu.");
        }
        return this.pdf(Matrix.columnVector(x));
    }

    private updatePDF(): void {
        const fac = Math.pow(Math.pow(2 * Math.PI, this.mu.columns) * this.detSigma, - 0.5);

        this.pdf = x => {
            if (!x.isColumnVector()) {
                throw new RangeError("x should be a column vector.");
            }
            const xCentred = Matrix.sub(x, this.mu);
            const val = Matrix.mul(xCentred.transpose(), Matrix.mul(this.invSigma, xCentred)).get(0, 0);
            return fac * Math.exp(- 0.5 * val);
        }
    }

    private mu: Matrix;

    // cached
    private detSigma: number;
    private invSigma: Matrix;

    private pdf: (x: Matrix) => number;
    
}

export function multivariateNormal(mean: Array<number>, cov: Array<Array<number>>) {
    return new MultivariateNormal(mean, cov);
}