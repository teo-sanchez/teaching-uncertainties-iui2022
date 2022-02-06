import { ModelOptions } from '@marcellejs/core';
import { GDA } from './gda.module';

export function gda(options?: Partial<ModelOptions>): GDA {
  return new GDA(options);
}

export type { GDA };
