import { PythonClassifier, PythonClassifierOptions } from './python-classifier.module';

export function pythonClassifier(options?: Partial<PythonClassifierOptions>): PythonClassifier {
  return new PythonClassifier(options);
}

export type { PythonClassifier, PythonClassifierOptions };
