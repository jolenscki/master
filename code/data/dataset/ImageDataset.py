from pathlib import Path
from typing import Any
from typing import Callable
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch
from torch.utils.data import Dataset

from util.h5_util import load_h5_file, write_data_to_h5
from util.data_range import generate_date_range, weekday_parser, date_parser
from util.graph_utils import reconstruct_graph, poor_man_graph_visualization
from util.graph_cell_structure import (offset_map, volume_channel_indices,
                                       speed_channel_indices)

# hard cap on test size
# we have 1 snapshot every 5 minutes, this equals to 12 snapshots/hour
# a test setup has duration of 2 hours; 1 hour for observing and 1 hour for
# guessing
# the start of the last would occur at the 22*12 = 264 > 256 = 2^8 snapshot
# this means that we wouldn't be able to store this value as uint8 (proposed
# as constraint in the challenge where the dataset comes from)
MAX_TEST_SLOT_INDEX = 240

def prepare_test(
    data: np.ndarray,
    offset=0,
    to_torch: bool = False
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    '''
    Function that separates the test data into observation and actual testing
    (meaning 1 hour observation and 6 snapshots of ground truth)
    
    Parameters
    ----------
    data: np.ndarray
        tensor of shape (24+, 495, 436, 8)
            24+: number of snapshots of city
            495, 436: dimension of city
            8: channels (volume and speed in 4 directions)
    offset: int
        offset from begining of data array
    to_torch: bool
        flag to determine if arrays to be returned are numpy or torch arrays
        
    Returns
    -------
    test_data: torch.float
        tensor of shape (12, 495, 436, 8) of consecutive observations
    ground_truth: torch.float
        tensor of shape (6, 495, 436, 8) of ground truth observations
    '''
    # generates array of indices of the ground_truth observations
    # equals to 5, 10, 15, 30, 45, and 60 minutes after test observations
    offsets = np.add([1, 2, 3, 6, 9, 12], 11 + offset)

    if isinstance(data, torch.Tensor):
        data = data.numpy()

    ub = offset + 12
    model_input = data[offset:ub]
    model_output = data[offsets]
    if to_torch:
        model_input = torch.from_numpy(model_input).float()
        model_output = torch.from_numpy(model_output).float()
    return model_input, model_output


class ImageDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        file_filter: Optional[str] = None,
        limit: Optional[int] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        '''
        Create pytorch Dataset from raw data files (*.h5), ImageDataset
        because we would be treating the grid as an image
        
        Parameters
        ----------
        root_dir
            data root folder, by convention should be `data/raw`, see `data/README.md`. All `**/training/*8ch.h5` will be added to the dataset.
        file_filter: str
            filter files under `root_dir`, defaults to `"**/training/*ch8.h5`
        limit
            truncate dataset size
        transform
            transform applied to both the input and label
        '''
        self.root_dir = root_dir
        self.limit = limit
        self.files = []
        self.file_filter = file_filter
        if self.file_filter is None:
            self.file_filter = "**/training/*8ch.h5"
        self.transform = transform
        # loads the dataset
        self._load_dataset()
        
    def _load_dataset(self):
        '''
        Internal method, recognizes the directory from ```root_dir``` and
        applies a regex filter (using ```rglob```), lists all files that can
        be found
        '''
        self.files = list(Path(self.root_dir).rglob(self.file_filter))
        
    def _load_h5_file(self, filename: str, sl: Optional[slice]) -> np.ndarray:
        '''
        Internal method that calls the ```load_h5_file```function
        
        Parameters
        ----------
        filename: str
            path to the file
        sl: slice
            slice to load
            
        Returns
        -------
        data: np.ndarray
            data loaded from file
        '''
        data = load_h5_file(file_path=filename, sl=sl)
        return data
    
    def __len__(self,):
        '''
        Private method, retrieves size of the Dataset
        
        Returns
        -------
        size: int
            size of the Dataset object
        '''
        # this calculates the len of the dataset; each file is hardcapped by 
        # the constant (see start of code to understand why)
        size = len(self.files) * MAX_TEST_SLOT_INDEX
        if self.limit is not None:
            return min(size, self.limit)
        return size
    
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        '''
        Private method, returns object in determined index. The idea is that
        given a certain index (representing the snapshot taken anytime from
        00:00 to 20:00), we would load the h5 files correspondant to a two
        hours interval. After that, ```prepare_test``` would separate the
        input (one hour observation) from output (5, 10, 15, 30, 45, 60 min.)
        and apply the transformations
        
        Parameters
        ----------
        idx: int
            index of object/datapoint to be accessed
            
        Returns
        -------
        input_data: torch.float
        output_data: torch.float
        '''
        if idx > self.__len__():
            raise IndexError("Index out of bounds")

        file_idx = idx // MAX_TEST_SLOT_INDEX
        start_hour = idx % MAX_TEST_SLOT_INDEX

        two_hours = self._load_h5_file(self.files[file_idx], sl=slice(start_hour, start_hour + 12 * 2 + 1))

        input_data, output_data = prepare_test(two_hours)

        input_data = self._to_torch(input_data)
        output_data = self._to_torch(output_data)

        if self.transform is not None:
            input_data = self.transform(input_data)
            output_data = self.transform(output_data)

        return input_data, output_data
    
    def _to_torch(self, data: np.ndarray) -> torch.float:
        '''
        Internal method that converts a numpy array to a torch array
        
        Parameters
        ----------
        data: np.ndarray
            data to be converted
        
        Returns
        -------
        data: torch.float
            array as torch float
        '''
        data = torch.from_numpy(data)
        data = data.to(dtype=torch.float)
        return data