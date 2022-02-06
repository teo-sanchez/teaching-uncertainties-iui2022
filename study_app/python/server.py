
import json
import asyncio
import websockets as ws
import janus


from typing import Any, Callable

from .trainer import Trainer, Predictor

from .import_common import *
from common.models import GMM, GDA, GaussianDensity

model = None
model_type = None

model_type_dict = {
    'gmm': lambda args: GMM(**args),
    'gda': lambda args: GDA(**args),
    'kernel': lambda args: GaussianDensity(**args)
}


async def monitor_training(q: janus.Queue, socket, trainer: Trainer, should_stop: Callable[[], bool]):
    global model
    while True:
        epoch_data = await q.get()
        await socket.send(json.dumps({
            'type': 'training_status',
            'data': epoch_data
        }))
        q.task_done()
        status = epoch_data['status']
        if status == 'success' or status == 'error':
            model = trainer.join()
            break
        elif should_stop():
            if status == 'idle':
                trainer.join()
                break
            else:
                trainer.stop()
                break


async def monitor_prediction(q: janus.Queue, socket, predictor: Predictor, should_stop: Callable[[], bool]):
    await q.get()
    result = predictor.join()
    await socket.send(json.dumps({
        'type': 'prediction',
        'data': result
    }))
    q.task_done()


async def main(socket, path: str):
    global model_type_dict
    global model_type
    global model
    should_stop_training = False
    should_stop_predicting = False
    async for s in socket:
        message = JSON.load_data(s)
        message_dict = json.loads(s)
        action = message.action
        data = message.data
        data_dict = message_dict['data']
        if action == 'train':
            should_stop_predicting = True
            should_stop_training = False
            queue = janus.Queue()
            model_type = data.model_type
            model = None
            model_func = model_type_dict[model_type]
            trainer = Trainer(
                model_func,
                model,
                data,
                data_dict,
                queue.sync_q
            )
            trainer.start()
            asyncio.create_task(
                monitor_training(
                    queue.async_q,
                    socket,
                    trainer,
                    lambda: should_stop_training
                )
            )
        elif action == 'predict':
            should_stop_predicting = False
            should_stop_training = True
            queue = janus.Queue()
            predictor = Predictor(model, data, queue.sync_q)
            predictor.start()
            asyncio.create_task(
                monitor_prediction(
                    queue.async_q,
                    socket,
                    predictor,
                    lambda: should_stop_predicting
                )
            )
        elif action == 'clear':
            model = None
            model_type = None
        elif action == 'stop':
            should_stop_training = True
            should_stop_predicting = True

if __name__ == '__main__':
    config = load_config()
    # dest_config = open_json_object(path.join(config_path, 'default.json'), 'r')
    #
    # url = f'{config.protocol}://{config.host}:{config.port}'
    # dest_url = f'http://{dest_config.host}:{dest_config.port}'
    start_server = ws.serve(main, config.host, config.port, max_size=None)
    print('Python server ready!')
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()
