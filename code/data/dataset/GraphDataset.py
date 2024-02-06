# Based on ```dataset_geometric.py``` from 
# https://github.com/iarai/NeurIPS2021-traffic4cast/blob/master/data/dataset/dataset_geometric.py
# Adapted
import os
from os import path as osp
import pickle
from multiprocessing import Pool
import tqdm
import h5py
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Optional
from typing import Tuple
from typing import Union
from typing import Dict

import matplotlib.pyplot as plt
import networkx as nx

import numpy as np
import torch
from torch import Tensor
import torch_geometric
from torch_geometric.data import Dataset

from util.h5_util import load_h5_file, write_data_to_h5
from util.data_range import generate_date_range, weekday_parser, date_parser
from util.graph_utils import reconstruct_graph, poor_man_graph_visualization
from util.graph_cell_structure import (offset_map, volume_channel_indices,
                                       speed_channel_indices)

from data.dataset import ImageDataset
from datetime import datetime, timedelta

# hard cap on test size
# we have 1 snapshot every 5 minutes, this equals to 12 snapshots/hour
# a test setup has duration of 2 hours; 1 hour for observing and 1 hour for
# guessing
# the start of the last would occur at the 22*12 = 264 > 256 = 2^8 snapshot
# this means that we wouldn't be able to store this value as uint8 (proposed
# as constraint in the challenge where the dataset comes from)
MAX_TEST_SLOT_INDEX = 240

class GraphDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        file_filter: Optional[str] = None,
        limit: Optional[int] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        pre_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        static_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        cities: Optional[list] = None,
        device: Optional[str] = None
    ):
        '''
        Create pytorch_geometric graph Dataset from raw data files (*.h5)
        
        Parameters
        ----------
        root_dir
            data root folder, by convention should be `data/raw`
        file_filter: str
            filter files under `root_dir`, defaults to `"**/training/*ch8.h5`
        limit
            truncate dataset size
        transform
            transform applied to both the input and label
        pre_transform
            transformation to be performed at the dynamical data
            for instance, collapsing 8 channels into 2
        static_transform
            transformation to be performed at the static data
        cities: Optional[list] (default = None)
            list of cities to be used in the dataset, if None use all cities
        device: Optional[str] (default = None)
            device in which the dataset will be loaded; if None, tries to load
            into cuda
        '''
        self.root_dir = root_dir
        self.graphs_dir = osp.join(osp.dirname(self.root_dir), 'graphs')
        self.limit = limit
        self.files = []
        self.device = device
        if self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.file_filter = file_filter
        if self.file_filter is None:
            self.file_filter = "**/training/*8ch.h5"
        self.cities = cities
        if self.cities is None:
            self.cities = os.listdir(self.root_dir)
        self.transform = transform
        self.pre_transform = pre_transform
        self.static_transform = static_transform
        self.static_identifier = self.get_identifier()
        super(GraphDataset, self).__init__(root_dir, transform, pre_transform)
        # loads the static_map for every city
        self.static_maps = self._create_static_maps_dict()
        # list the files that compose the dataset
        self._list_files()
        # creates/loads city_graphs
        self.city_graphs = self._generate_city_graphs()

    
    
    def _list_files(self):
        '''
        Internal method, recognizes the directory from ```root_dir``` and
        applies a regex filter (using ```rglob```), lists all files that can
        be found
        '''
        files = list(Path(self.root_dir).rglob(self.file_filter))
        # second level of filter: only cities stated on the init method
        self.files = []
        for f in files:
            city = osp.basename(f).split("_")[1]
            if city in self.cities:
                self.files.append(f)
        
    def _load_h5_file(self, filename: str, sl: Optional[slice]=None) -> np.ndarray:
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
    
    def _load_static_map(self, filename: str) -> np.ndarray:
        '''
        Internal method that loads the static map file `<CITY NAME>_static.h5`
        
        Parameters
        ----------
        filename: str
            path to the file
            
        Returns
        -------
        data: np.ndarray
            static array of the city; tensor of shape (9, 495, 436)
            first channel stands for a low res representation of the city
            every other channel is a binary representation of the connection
            between each region and, respectively, the following directions
            'N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'
        '''
        static_map = load_h5_file(file_path=filename)
        if self.static_transform is not None:
            static_map = self.static_transform(static_map)
        return static_map
    
    def _create_static_maps_dict(self,) -> Dict[str, np.ndarray]:
        '''
        Internal method that creates a dictionary with the city names as keys
        and the static maps of the cities as values
        
        Returns
        -------
        static_dict: Dict[str, np.ndarray]
        '''
        # first we verify the folders of the root_dir; each subfolder is
        # supposed to be a city
        static_dict = {}
        for city in self.cities:
            path = osp.join(self.root_dir, city)
            filt = "*_static.h5"
            filename = next(Path(path).rglob(filt))
            static_map = self._load_static_map(filename)
            static_dict[city] = static_map
        return static_dict
    
    def _generate_city_graphs(
        self,
        cities: Optional[list] = None,
        save: Optional[bool] = False
        ) -> Dict:
        '''
        Private method that generates a graph for every city on the dataset
        and stores the graph as a dict
        
        Parameters
        ----------
        cities: Optional (default = None)
            list of cities to generate the city graphs, if None generates for 
            all cities in root_dir
        save: Optional (default = False)
            bool flag that indicates whether the graphs should be saved as pkl
            files or not; if False: don't save
            
        Returns
        -------
        city_graphs: Dict[str, Dict[]]
            graph of cities, every key is the name of the city in UPPER CASE
            every value is a dict with the graph and the mapping scheme
            (node ID to node)
        '''
        if cities is None:
            cities = list(self.static_maps.keys())
        
        city_graphs: Dict[str, Dict] = {}
        # now we actually create the graphs
        for city in cities:
            # check if we already have this city saved
            path = osp.join(self.graphs_dir, f'{city}_graph.pkl')
            if osp.exists(path) and self.check_identifier():
                with open(path, 'rb') as pkl:
                    city_graph = pickle.load(pkl)
                city_graphs[city] = city_graph
                continue
            city_graphs[city]: Dict = {}
            static_map = self.static_maps[city]
            max_row, max_col = static_map.shape[1:]
            G = nx.Graph(name=city)
            offsets = list(offset_map.values())
            
            # for each direction ('N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW')
            # calculate conectivities
            # starts at 1 (first axis is city representation)
            for i in range(1, 9):
                start_coordinates = np.argwhere(static_map[i, ...] > 0)
                # transform into list of tuple
                start_coordinates = [(row, column) for row, column in start_coordinates]
                # filter start_coordinates to get rid of nodes with negative
                # numbers or numbers large than the size of the map
                start_coordinates = list(
                    filter(
                        lambda x: x[0] >= 0 and x[0] < max_row and x[1] >= 0 and x[1] < max_col, start_coordinates
                        )
                    )
                # i - 1 since offsets has len = 8
                dr, dc = offsets[i - 1]
                end_coordinates = [(row + dr, column + dc) for row, column in start_coordinates]
                # filter end_coordinates to get rid of nodes with negative 
                # numbers or numbers large than the size of the map
                end_coordinates = list(
                    filter(
                        lambda x: x[0] >= 0 and x[0] < max_row and x[1] >= 0 and x[1] < max_col, end_coordinates
                        )
                    )
                G.add_edges_from(zip(start_coordinates, end_coordinates))
            
            # now we perform a mapping to assign an ID to every node in the graph
            mapping = {node: idx for idx, node in enumerate(G.nodes())}
            
            # now we create a connection matrix using the indexes of the nodes
            # that share an edge
            # so if nodes (of index) 1 and 2, 1 and 3, 5 and 230 share an edge
            # edge_idx = [[1, 1,   5]
            #             [2, 3, 230]]
            edge_idx: Tensor = torch.tensor([
                [mapping[n] for n, _ in G.edges],
                [mapping[n] for _, n in G.edges]
            ], dtype=torch.long)
            # replicate the pairs so that we have two edges for each connection
            # 0 -> 1 and 1 -> 0 being held in different parts of this matrix
            edge_idx = torch.cat((edge_idx, edge_idx[[1, 0]]), axis=1)
            city_graphs[city] = {'edge_idx':   edge_idx,
                                 'mapping' :    mapping,
                                 'G'       :          G,
            }
            
            if save:
                # check if we have a directory for the processed data/files
                # it should be locate at data/graphs
                if not osp.isdir(self.graphs_dir):
                    Path(self.graphs_dir).mkdir(parents=True, exist_ok=True)
                # check if the graphs aren't already there
                city_graph_path = osp.join(self.graphs_dir, f'{city}_graph.pkl')
                if not osp.isdir(city_graph_path):
                    with open(city_graph_path, 'wb') as pkl:
                        pickle.dump(city_graphs[city], pkl)
        
        # save the static transform identifier
        if save:
            path = osp.join(self.graphs_dir, 'static_identifier.pkl')
            with open(path, 'wb') as pkl:
                pickle.dump(self.static_identifier, pkl)

        return city_graphs

    def get_identifier(self,):
        '''
        Returns a unique identifier for the graph based on the square index
        and size of the static transform.
        
        Returns
        -------
        str
            A string representing the unique identifier for the graph.
        
        Notes
        -----
        If no static transform is available, an empty string is returned.
        '''
        static_transform = self.static_transform
        if static_transform is None:
            return "None"
        sqr_idx = static_transform.transforms[0].square_index
        sqr_size = static_transform.transforms[0].square_size
        return f"{sqr_size}_{sqr_idx}"
            
    def check_identifier(self,):
        '''
        Method that checks if the static identifier of the dataset is the same
        as the one saved in the graphs directory
        
        Returns
        -------
        bool
            True if the static identifier is the same, False otherwise
        '''
        path = osp.join(self.graphs_dir, 'static_identifier.pkl')
        if not osp.exists(path):
            return False
        with open(path, 'rb') as pkl:
            static_identifier = pickle.load(pkl)
        return static_identifier == self.static_identifier
    
    def process_dynamic_data(self, filename, save_city_graphs=False) -> np.ndarray:
        '''
        Method that processes the dynamic data contained in a file
        
        Parameters
        ----------
        filename: str
            full path to the raw file (from self.files)
        save_city_graphs: Optional[bool] (default = False)
            bool flag that indicates whether the graphs should be saved as pkl
            
        Returns
        -------
        dynamic_data: np.ndarray
            data (converted from the raw data with the static method
            ```image_to_graph```
        '''
        city = osp.basename(filename).split("_")[1]
        if city not in self.city_graphs:
            new_city_graph = self._generate_city_graphs([city], save=save_city_graphs)
            self.city_graphs = dict(self.city_graphs, **new_city_graph)

        city_graph = self.city_graphs[city]
        output_path = osp.join(self.graphs_dir, city, osp.basename(filename))

        # check if this file has been processed
        if osp.exists(output_path) and self.check_identifier():
            with h5py.File(output_path, 'r') as output_h5:
                dynamic_data = np.array(output_h5.get("array"))
            return dynamic_data

        dynamic_data = self._load_h5_file(filename)
        if self.pre_transform is not None:
            dynamic_data = self.pre_transform(dynamic_data)
        
        dynamic_data = self.image_to_graph(city_graph, dynamic_data)
        Path(osp.join(self.graphs_dir, city)).mkdir(parents=True, exist_ok=True)
        num_channels = dynamic_data.shape[-1]
        with h5py.File(output_path, 'w') as output_h5:
            output_h5.create_dataset(
                "array",
                shape=dynamic_data.shape,
                chunks=(1, dynamic_data.shape[1], num_channels),
                dtype="uint8",
                data=dynamic_data,
                compression="lzf"
            )
            
        
        return dynamic_data
    
    def batch_process(self):
        '''
        Method for processing all files in self.files, to be used only once
        '''
        for f in self.files:
            self.process_dynamic_data(f)
        
    @staticmethod    
    def image_to_graph(city_graph, dynamic_data) -> np.ndarray:
        '''
        Function that transforms a dynamic_data array (image of the grid of 
        the city) into a graph array
        
        Parameters
        ----------
        city_graph: Dict[str, _]
            dict containing the index of the edges of the graph, the mapping
            array (node ID to node) and the graph object
        dynamic_data: np.ndarray
            tensor of shape (24+, 496, 435, 8) containing the 8 channels' grid
            of the city
            
        Returns
        -------
        x: np.ndarray
            tensor of shape (24+, len(nodes), 8)
        '''
        edge_idx, mapping, G = city_graph.values()
        nodes = np.array(list(mapping.keys()))
        x = dynamic_data[:, nodes[:, 0], nodes[:, 1], :]
        
        return x
    
    def graph_to_image(
        self,
        city: str, 
        graph_data: torch.Tensor,
        ) -> torch.Tensor:
        '''
        Function that transforms a graph array into a dynamic_data array (image
        of the grid of the city)

        Parameters
        ----------
        city: str
            name of the city
        graph_data: torch.Tensor
            tensor of shape (BATCH_SIZE*len(nodes), num_channels, seq_len) 
            containing the data of the graph

        Returns
        -------
        dynamic_data: torch.Tensor
            tensor of shape (batch_size, seq_len, height, width, num_channels)
        '''
        assert city in self.city_graphs, f"City {city} not in self.city_graphs"
        graph_data = graph_data.to(self.device)
        city_graph = self.city_graphs[city]
        _, mapping, _ = city_graph.values()

        # Determine the grid dimensions
        height, width = self.static_maps[city].shape[1:3]
        num_nodes = len(mapping)
        num_channels, seq_len = graph_data.shape[1], graph_data.shape[2]
        batch_size = graph_data.shape[0]//num_nodes

        # Reshape and reorder graph_data
        graph_data = graph_data.view(batch_size, num_nodes, num_channels, seq_len)
        graph_data = graph_data.permute(0, 3, 1, 2)  # now shape is (batch_size, seq_len, num_nodes, num_channels)

        # Initialize dynamic_data tensor
        dynamic_data = torch.zeros(
            (batch_size, seq_len, height, width, num_channels),
            dtype=graph_data.dtype
        ).to(self.device)

        # Map each node's data back to its position in the grid
        for (row, col), node_id in mapping.items():
            dynamic_data[:, :, row, col, :] = graph_data[:, :, node_id, :]

        return dynamic_data

    
    def create_data_object(
        self,
        x: np.ndarray,
        y: np.ndarray,
        edge_index: torch.Tensor,
        date_features: np.ndarray
        ) -> torch_geometric.data.Data:
        '''
        Method that converts a numpy array representation of the graph into a
        Data object (to be used in torch_geometric)
        
        Parameters
        ----------
        x: np.ndarray
            array of shape (12, len(nodes), 8) with the `training` part of the
            data, input
        y: np.ndarray
            array of shape (6, len(nodes), 8) with the ground truth of the
            data, composed of snapshots of 5, 10, 15, 30, 45, and 60 minutes
        edge_index: torch.Tensor
            tensor that describes the connectivity of the graph
        date_features: np.ndarray
            array of shape (4,) with the date features (sine-cosine encoding)
            for the given file index and start hour considering day-of-the-week
            and hour-of-the-day
            
        Returns
        -------
        data_obj: torch_geometric.data.Data
            data object describing a homogeneous graph
        '''
        # we have 12 snapshots 5 minutes apart
        input_len    = x.shape[0]
        num_channels = x.shape[-1]
        num_nodes    = x.shape[1]
        output_len   = y.shape[0]
        
        # normalize input and output array from 0-255 to 0-1
        x = torch.from_numpy(x)/255
        y = torch.from_numpy(y)
        
        # change order of dimensions
        # x's shape becomes (len(nodes), 12, 8)
        # y's shape becomes (len(nodes),  6, 8)
        x = torch.moveaxis(x, 1, 0)
        y = torch.moveaxis(y, 1, 0)
        
        # reshape x and y so that
        # x's shape becomes (len(nodes), 12*8)
        # y's shape becomes (len(nodes),  6*8)
        x = x.reshape(num_nodes,  input_len*num_channels)
        y = y.reshape(num_nodes, output_len*num_channels)

        data_obj = torch_geometric.data.Data(
            x=x.float(),
            y=y.float(),
            edge_index=edge_index,
            date_features=torch.from_numpy(date_features).float()
        ).to(self.device)
        
        return data_obj
    
    def len(self,):
        '''
        Private method, retrieves size of the Dataset
        
        Called from `__getitem()__` from parent class!
        
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
    
    def get(self, idx: int) -> torch_geometric.data.Data:
        '''
        Private method, returns object in determined index. The idea is that
        given a certain index (representing the snapshot taken anytime from
        00:00 to 20:00), we would load the h5 files correspondant to a two
        hours interval. We process the dynamic data (using 
        ```process_dynamic_data```), which loads the dynamic data and
        transforms it to a graph structure. Afterwards we use this output to
        create a torch_geometric Datas object (that will be effectivelly the
        input for the network)
        
        Parameters
        ----------
        idx: int
            index of object/datapoint to be accessed
            
        Returns
        -------
        data: torch_geometric.data.Data
        '''
        if idx > self.len():
            raise IndexError("Index out of bounds")

        file_idx = idx // MAX_TEST_SLOT_INDEX
        start_hour = idx % MAX_TEST_SLOT_INDEX
        
        filename = self.files[file_idx]
        dynamic_data = self.process_dynamic_data(filename)
        city = osp.basename(filename).split("_")[1]
        city_graph = self.city_graphs[city]
        
        x = dynamic_data[start_hour:start_hour + 12, :, :]
        offsets = np.add([1, 2, 3, 6, 9, 12], 11 + start_hour)
        y = dynamic_data[offsets, :, :]
        edge_idx = city_graph['edge_idx']

        date_features = self.date_embedding(file_idx, start_hour)
        
        data_obj = self.create_data_object(x, y, edge_idx, date_features)

        return data_obj
    
    @staticmethod
    def date_embedding(file_idx, start_hour):
        """
        Calculates the sine-cosine encoding for the given file index and start hour.

        Parameters
        ----------
        file_idx : int
            The index of the file, represents the day.
        start_hour : int
            The starting hour.

        Returns
        -------
        np.array
            The encoded time features as a numpy array.
        """
        start_date = datetime(2019, 1, 2)
        current_date = start_date + timedelta(days=file_idx, hours=start_hour)


        hours_in_day = MAX_TEST_SLOT_INDEX // 12
        days_in_week = 7

        # Sine-cosine encoding for hour of the day
        hour_sin = np.sin(2 * np.pi * current_date.hour / hours_in_day)
        hour_cos = np.cos(2 * np.pi * current_date.hour / hours_in_day)

        # Sine-cosine encoding for day of the week
        day_sin = np.sin(2 * np.pi * current_date.weekday() / days_in_week)
        day_cos = np.cos(2 * np.pi * current_date.weekday() / days_in_week)

        # Combine into a single feature vector
        time_features_periodic = np.array([hour_sin, hour_cos, day_sin, day_cos])

        return time_features_periodic
