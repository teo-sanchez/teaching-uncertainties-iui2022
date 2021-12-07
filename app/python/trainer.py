
import os
import janus

from threading import Thread
from typing import Callable, Any
from types import SimpleNamespace
import numpy as np


class Trainer(Thread):
    def __init__(self, model_gen_func: Callable, model, data: object, data_dict: object, queue: janus.Queue):
        Thread.__init__(self)
        self.model = model
        self.model_gen_func = model_gen_func
        self.data = data
        self.data_dict = data_dict
        self.queue = queue
        self.terminated = False
        self._return = None

    def stop(self):
        self.terminated = True

    def run(self):
        if self.model is None:
            self.queue.put({'status': 'loading'})
            parameters = self.data_dict['parameters']
            self.model = self.model_gen_func(parameters)
        ins = self.data.training_data
        self.queue.put({'status': 'epoch'})
        self.model.fit(np.array(ins.z), np.array(ins.y))
        self._return = self.model
        self.queue.put({'status': 'success'})

    def join(self, *args):
        Thread.join(self, *args)
        return self._return


class Predictor(Thread):
    def __init__(self, model, data: Any, queue: janus.Queue):
        Thread.__init__(self)
        self.model = model
        self.data = data
        self.queue = queue
        self.terminated = False
        self._return = None
        self.queue.put({'status': 'done'})

    def stop(self):
        self.terminated = True

    def run(self):
        if self.model is None:
            self._return = {}
            self._return['densities'] = [None]
            return
        ins = self.data.prediction_data
        z = np.array(ins.z)
        distances = self.model.distance(z)
        self._return = {}
        self._return['distances'] = distances.tolist(),
        self._return['scaler_parameters'] = self.model.get_scaler_params()
        self._return['status'] = 'success'

    def join(self, *args):
        Thread.join(self, *args)
        return self._return
