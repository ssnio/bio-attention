# # built-in modules
import os
import json
import time
import logging
import csv
from logging import Logger
# # Torch modules
import torch


def setup_logger(logger_dir: str, logger_name: str, logger_id: str):
    """
    Args:
        logger_dir (str): Directory to save the log file into
        logger_name (str): Name of the log file
        logger_id (str): Unique identifier for the log file
    """
    # file and directory
    filename = os.path.join(logger_dir, logger_name + "_" + logger_id + ".log")
    logger = logging.getLogger(logger_name)

    # Set up handlers if not done already
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(module)s:%(funcName)s - %(message)s')
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    return logger


def startup_folders(dir: str, name: str = None):
    name = name if name is not None else "exp_"

    # # make directory
    if not os.path.exists(dir):
        os.makedirs(dir)
        print(f"{dir} was created!")

    # # create folder for results
    folder_id = int(time.time())
    results_folder = os.path.join(dir, str(folder_id))
    while os.path.exists(results_folder):
        folder_id += 1
        results_folder = os.path.join(dir, str(folder_id))
    os.makedirs(results_folder)
    print(f"{results_folder} was created!")

    # # setup logger
    logger = setup_logger(results_folder, name, str(folder_id))
    return results_folder, logger

def save_dicts(dicts, dir, name, logger):
    for k, v in dicts.items():
        if isinstance(v, dict):
            for q, p in v.items():
                try:
                    json.dumps(p)
                except (TypeError, OverflowError):
                    if hasattr(p, "__name__"):
                        v[q] = p.__name__
                    elif q == "datasets":
                        v[q] = tuple([d.__class__.__name__ for d in p])
                    elif q == "dataloaders":
                        v[q] = tuple([None for d in p])
                    else:
                        logger.info(f"Could not pass {q} value of {p} to JSON.")
                        v[q] = str(p)
        try:
            json.dumps(v)
        except (TypeError, OverflowError):
            if hasattr(v, "__name__"):
                dicts[k] = v.__name__
            else:
                logger.info(f"Could not pass {k} value of {v} to JSON.")
                dicts[k] = str(v)

    with open(os.path.join(dir, name + ".json"), "w") as json_file:
        json.dump(dicts, json_file)


def load_dicts(dir, name):
    with open(os.path.join(dir, name + ".json"), "r") as json_file:
        dicts = json.load(json_file)
    for k, v in dicts.items():
        if v == 'ReLU()':
            dicts[k] = torch.nn.ReLU()
        elif v == 'Tanh()':
            dicts[k] = torch.nn.Tanh()
        elif v == 'GELU()' or v == "GELU(approximate='none')":
            dicts[k] = torch.nn.GELU()
        elif v == 'Identity()':
            dicts[k] = torch.nn.Identity()
        elif v == "(GELU(approximate='none'), ReLU())":
            dicts[k] = (torch.nn.GELU(approximate='none'), torch.nn.ReLU())
    return dicts


def get_device():
    # # set device preferably to GPU
    num_workers, pin_memory = 0, False
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # NVIDIA GPU
        num_workers, pin_memory = 4, True
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")  # Apple Silicon (Metal)
    else:
        device = torch.device("cpu")
    print(f"Device set to {device}")
    return device, num_workers, pin_memory


def save_results_to_csv(data: list, filename: str, header: list, logger: Logger):
    header = list(header[i] for i in range(len(data)) if len(data[i]) > 0)
    data = list(data[i] for i in range(len(data)) if len(data[i]) > 0)
    data = list(zip(*data))
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        if header is not None:
            writer.writerow(header)
        writer.writerows(data)
    file.close()
    logger.info(f"Results saved to {filename}")
