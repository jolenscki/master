import glob
import os
import os.path as osp
import numpy as np

def align_datasets(ds_dict):
    # Find the size of the largest dataset
    max_size = max(len(ds) for ds in ds_dict.values())

    # Resample smaller datasets to match the largest dataset
    for city, dataset in ds_dict.items():
        if len(dataset) < max_size:
            # Calculate the number of times to replicate the dataset
            repetitions = max_size // len(dataset)
            
            # Create a concatenated dataset
            new_dataset = dataset
            for _ in range(repetitions - 1):
                new_dataset = new_dataset.concatenate(dataset)
            
            # If there are remaining samples to add, concatenate them
            remaining_samples = max_size - len(new_dataset)
            if remaining_samples > 0:
                new_dataset = new_dataset.concatenate(dataset[:remaining_samples])

            # Update the dataset in the dictionary
            ds_dict[city] = new_dataset

    return ds_dict


def logging(level: int=0, message: str=''):
    '''
    Prints a log message with a given indentation level.

    Parameters
    ----------
    level : int
        The indentation level of the log message.
    message : str
        The message to be logged.

    Returns
    -------
    None
    '''
    if level == 0:
        return
    else:
        print(f'{"    "*level} {message}')

def generate_log_name(prefix="", fdir="training logs"):
    """
    Generates a unique log file name with the given prefix in the specified directory.

    Parameters
    ----------
    prefix : str, optional
        The prefix to use for the log file name. Defaults to an empty string.
    fdir : str, optional
        The directory to search for existing log files. Defaults to "training logs".

    Returns
    -------
    str
        A unique log file name with the given prefix in the specified directory.
    """
    files = glob.glob(fdir + f'/{prefix}*')
    return prefix + "_" + str(len(files)).zfill(4) + ".log"

def check_dir(dir):
    """
    Create a directory if it doesn't exist.

    Parameters
    ----------
    dir : str
        The directory path.

    Returns
    -------
    None

    Examples
    --------
    >>> check_dir('/path/to/directory')
    """
    if not osp.exists(dir):
        os.makedirs(dir)