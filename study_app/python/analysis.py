
import re
import os
import argparse

from .import_common import *
from common.utilities import JSON

# todo: everything


def recover_participant_data(path: str, name: str, condition: str):
    # get all files in path (non-recursive).
    # from: https://stackoverflow.com/a/3207973
    files = [f for f in os.listdir(
        path) if os.path.isfile(os.path.join(path, f))]
    # match for name and condition
    pass


def recover_participant_models(
    model_db: object,
    root_path: str,
    id: str,
    condition: str
):
    # todo: load from config
    ensemble_size = 3

    # also get the total budget
    # (number of times the ensemble has been trained)
    total_budget = -1
    #
    model_data = []
    for (idx, entry) in enumerate(model_db):
        # entry names are separated by low dashes.
        info = entry['name'].split('_')
        files = entry['files']
        # participant and conditions
        participant_id, condition = info[0].split('-')
        if participant_id == id:
            data = {}
            budget = info[1]
            #
            if total_budget < budget:
                total_budget = budget
            #
            model_idx = info[3]
            data.budget = budget
            data.model_idx = model_idx

            data.path = files[1]

            model_data.append(data)

    # those should be the same,
    # if they aren't
    assert(total_budget == len(model_data) / 3)

    model_weights_path = [None] * ensemble_size
    for i in range(len(model_weights_path)):
        model_weights_path = [None] * total_budget
    for data in model_data:
        model_weights_path[data.model_idx][data.budget] = data.path
        # get all files in path (non-recursive).
        # from: https://stackoverflow.com/a/3207973
    # model_dirs = [f for f in os.listdir( root_path) if os.path.isdir(os.path.join(root_path, f))]
    # match for name and condition
    model_info = {}
    model_info.id = id
    model_info.condition = condition
    model_info.weights = model_weights_path
    return model_info


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract model weights for the participant")
    parser.add_argument('--root', metavar='path',
                        required=True, help='Path to TFJS model directory')
    parser.add_argument('--db', metavar='path',
                        required=True, help='Path to TFJS db file')
    parser.add_argument('--id', metavar='string',
                        required=True, help='Id of the participant')
    parser.add_argument('--condition', metavar='string',
                        required=True, help='Condition')
    args = parser.parse_args()
    model_info = recover_participant_models(
        args.db, args.root, args.id, args.condition)
    print(model_info.weights[0][3])
