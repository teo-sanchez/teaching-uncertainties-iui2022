import { 
  Dataset, 
  ObjectId, 
  Parametrable,
  StoredModel,
  Model,
  ModelOptions,
  logger,
  TrainingStatus,
} from '@marcellejs/core';

import { DensityEstimatorResults } from '../../../types';

export interface PythonClassifierOptions extends Partial<ModelOptions> {
  modelType: 'gmm' | 'gda' | 'kernel',
  parameters: {
    [key: string]: any
  }
}

interface DeferredPromise {
  promise: Promise<any>;
  resolve: (...args: any[]) => void;
  reject: () => void;
}

export class PythonClassifier extends Model<number[][], DensityEstimatorResults> {

  title = 'Gaussian Mixture Model';
  options: PythonClassifierOptions;

  private socket: WebSocket;

  private hasBeenTrained: boolean;
  private labels: Array<string>;

  private trainPromise: DeferredPromise
  private predictPromise: DeferredPromise

  constructor(options?: Partial<PythonClassifierOptions>) {
    super(options);
    this.options = {
      modelType: 'gmm',
      parameters: { },
      ...options
    };
  }

  parameters: Parametrable['parameters'];
  serviceName: 'python-classifier';

  async connect(url: string): Promise<void> {
    const reconnectTimeOutInit = 1000;
    const reconnectIncrement = 5000; // ms
    //
    var reconnectTimeOut = reconnectTimeOutInit; // ms
    return new Promise((resolve, reject) => {
      logger.info(`Connecting to (python) backend ${url}...`);
      this.socket = new WebSocket(url);
      this.socket.onerror = (ev: Event) => {
        logger.error(`Failed to connect to (python) backend ${url}.`);
        reconnectTimeOut += reconnectIncrement;
        return reject(ev);
      };
      this.socket.onopen = () => {
        reconnectTimeOut = reconnectTimeOutInit; // ms
        logger.info(`Connected to (python) backend ${url}!`);
        return resolve();
      };
      // reconnect when closed.
      // see: https://stackoverflow.com/a/23176223
      this.socket.onclose = (event) => {
        logger.info(`Disconnected from (python) backend ${url}.`);
          this.hasBeenTrained = false;
        setTimeout(() => { 
          this.connect(url); 
        }, reconnectTimeOut)
      }
    });
  }

  async reset(): Promise<void> {
    // todo await an ack
    this.socket.send(
      JSON.stringify({
        action: 'reset',
        data: null
      })
    );
  }

  // @Catch
  async train(dataset: Dataset<number[][], string>): Promise<void> {
    const instances = await dataset.items().toArray();
    const zs = instances.map(inst => inst.x.flat());
    const ys = instances.map(inst => inst.y);
    this.labels = Array.from(new Set(ys));
    if (this.predictPromise) {
      // stop predicting
      this.predictPromise.resolve(null);
    }
    const promise = new Promise<void>((resolve, reject) => {
      // taking resolve and reject outside of Promise
      // so that we can stop it elsewhere.
      // see: https://stackoverflow.com/a/36072263
      this.trainPromise = { ...this.trainPromise, resolve, reject };
      if (!this.socket) {
        reject('No socket connection');
      }
      this.socket.onmessage = (ev) => {
        const { type, data } = JSON.parse(ev.data);
        if (type === 'training_status') {
          const status = data as TrainingStatus;
          this.$training.set(status);
          switch (status.status) {
            case 'success':
              this.hasBeenTrained = true;
              this.trainPromise = null;
              return resolve();
            case 'error':
              this.hasBeenTrained = false;
              this.trainPromise = null;
              return reject();
            default: // 'epoch' or other
              break;
          }
        }
      };
      const data = {
        training_data: {
          z: zs,
          y: ys.map(y => this.labels.indexOf(y)),
        },
        parameters: {
          ...this.options.parameters
        },
        model_type: this.options.modelType,
      };
      this.socket.send(
        JSON.stringify({
          action: 'train',
          data
        })
      );
    });
    this.trainPromise = { ...this.trainPromise, promise };
    return promise;
  }

  async predict(z: number[][]): Promise<DensityEstimatorResults> {
    if (!this.hasBeenTrained || this.trainPromise) {
      // wait for training to finish
      return null;
    }
    const promise = new Promise<DensityEstimatorResults>((resolve, reject) => {
      // taking resolve and reject outside of Promise
      // so that we can stop it elsewhere.
      // see: https://stackoverflow.com/a/36072263
      this.predictPromise = { ...this.predictPromise, resolve, reject };
      if (!this.socket) {
        reject('No socket connection');
      }
      this.socket.onmessage = (ev) => {
        const { type, data } = JSON.parse(ev.data);
        if (type === 'prediction') {
          const { distances, scaler_parameters, ...rest } = data;
          const distance = distances[0][0] as number;
          const result = {
            distance: distance,
            scalerParameters: scaler_parameters,
            ...rest
          } as DensityEstimatorResults;
          this.predictPromise = null;
          return resolve(result);
        }
      };
      const data = {
        prediction_data: {
          z: z,
        }
      }
      this.socket.send(
        JSON.stringify({
          action: 'predict',
          data
        })
      );
    });
    this.predictPromise = {
      ...this.predictPromise,
      promise
    };
    return promise;
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
}
