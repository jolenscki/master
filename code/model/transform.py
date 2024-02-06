import numpy as np
from typing import Union, List, Tuple, Optional

class CollapseChannels(object):
    '''
    Transformation that collapse the channels in our data. Since the even
    channels (0, 2, 4, 6) represent the volume (or flow) of vehicles in each
    of the 4 main diagonals, and the odd channels (1, 3, 5, 7) represent the 
    speed of displacement of vehicles in, again, each of the 4 main diagonals,
    we aim here to just mean all flows and speeds
    '''
    def __call__(self, dynamic_data):
        '''
        Parameters
        ----------
        dynamic_data: np.ndarray
            tensor of shape (24+, 496, 435, 8)
        
        Returns
        -------
        collapsed_data: np.ndarray
            tensor of shape (24+, 496, 435, 2)
        '''
        # Calculate the mean for odd and even channels
        mean_odd = dynamic_data[..., 1::2].mean(axis=-1, keepdims=True)
        mean_even = dynamic_data[..., ::2].mean(axis=-1, keepdims=True)

        # Concatenate the results to create new channels
        collapsed_data = np.concatenate((mean_even, mean_odd), axis=-1)
        return collapsed_data


class ExtractSquare(object):
    '''
    Transformation that extracts a square from the dynamic data. This is useful to
    reduce the size of the data, and also to reduce the number of nodes in the
    graph

    Parameters
    ----------
    square_size: Union[int, List[int]]
        size of the square to extract. If int, then the square will be of size
        (square_size, square_size). If List[int], then the square will be of
        size (square_size[0], square_size[1])
    square_index: Optional[Union[str, int, List[int]]]
        index of the square to extract. If int, then the square will be the
        square_index-th square in the grid. If List[int], then the square will
        be the square with the row index square_index[0] and the column index
        square_index[1]. If 'central', then the square will be the central
        square in the grid
    shape_hint: Tuple[int, int]
        shape of the original data. This is used to calculate the number of
        rows and columns in the grid
    '''
    def __init__(self, square_size, square_index, shape_hint=(496, 435)):
        if isinstance(square_size, int):
            self.square_size = [square_size, square_size]
        else:
            self.square_size = square_size

        self.nrows = shape_hint[0] // self.square_size[0]
        self.ncols = shape_hint[1] // self.square_size[1]
        if isinstance(square_index, str):
            if square_index == 'central':
                self.square_index = self.get_central_square_index(self.nrows, self.ncols)
            else:
                raise ValueError('square_index must be an int or "central"')
        elif isinstance(square_index, list):
            self.square_index = self.get_square_index_from_coordinates(
                self.nrows, self.ncols, square_index[0], square_index[1]
            )
        else:
            self.square_index = square_index

    def __call__(self, dynamic_data):
        '''
        Parameters
        ----------
        dynamic_data: np.ndarray
            tensor of shape (24+, 496, 435, 2)

        Returns
        -------
        square: np.ndarray
            tensor of shape (24+, p, q, 2)
        '''
        # transform shape (24+, 496, 435, 2) -> (496, 435, 2, 24+)
        x = np.moveaxis(dynamic_data, 0, -1)

        # split on rows
        split1 = np.array_split(
            x,
            range(self.square_size[0], x.shape[0], self.square_size[0]),
            axis=0
            )

        # split on columns
        split2 = [
            np.array_split(
                chunk, 
                range(self.square_size[1], chunk.shape[1], self.square_size[1]), 
                axis=1
            ) for chunk in split1
        ]

        # flatten the list
        squares = [item for sublist in split2 for item in sublist]

        # Discard any squares that are not of size (p, q)
        squares = [sq for sq in squares if list(sq.shape[:2]) == self.square_size]
        
        square = np.array(squares[self.square_index])
        # move dimensions back to place
        square = np.moveaxis(square, -1, 0)

        return square
    
    @staticmethod
    def get_central_square_index(nrows, ncols):
        """
        Returns the index of the central square in a 2D grid represented as a 1D array.

        Parameters
        ----------
        nrows : int
            The number of rows in the 2D grid.
        ncols : int
            The number of columns in the 2D grid.

        Returns
        -------
        int
            The index of the central square in the 1D array representation of the grid.
        """
        central_row = nrows // 2
        central_col = ncols // 2
        return central_row * ncols + central_col

    @staticmethod
    def get_square_index_from_coordinates(
        nrows: int, ncols: int, row: int, col: int
        ) -> int:
        """
        Returns the index of the square in a 2D grid represented as a 1D array.

        Parameters
        ----------
        nrows : int
            The number of rows in the 2D grid.
        ncols : int
            The number of columns in the 2D grid.
        row : int
            The row index of the square.
        col : int
            The column index of the square.

        Returns
        -------
        int
            The index of the square in the 1D array representation of the grid.
        """
        return row * ncols + col

# LEGACY: FUNCTIONS

def collapse_channels(dynamic_data: np.ndarray) -> np.ndarray:
    '''
    Function that collapse the channels in our data. Since the even channels
    (0, 2, 4, 6) represent the volume (or flow) of vehicles in each of the 4
    main diagonals, and the odd channels (1, 3, 5, 7) represent the speed of
    displacement of vehicles in, again, each of the 4 main diagonals, we aim
    here to just mean all flows and speeds

    Parameters
    ----------
    dynamic_data: np.ndarray
        tensor of shape (24+, 496, 435, 8)
    
    Returns
    -------
    collapsed_data: np.ndarray
        tensor of shape (24+, 496, 435, 2)
    '''
    # Calculate the mean for odd and even channels
    mean_odd = dynamic_data[..., 1::2].mean(axis=-1, keepdims=True)
    mean_even = dynamic_data[..., ::2].mean(axis=-1, keepdims=True)

    # Concatenate the results to create new channels
    collapsed_data = np.concatenate((mean_even, mean_odd), axis=-1)
    return collapsed_data

def get_central_square_index(nrows, ncols):
    """
    Returns the index of the central square in a 2D grid represented as a 1D array.

    Parameters
    ----------
    nrows : int
        The number of rows in the 2D grid.
    ncols : int
        The number of columns in the 2D grid.

    Returns
    -------
    int
        The index of the central square in the 1D array representation of the grid.
    """
    central_row = nrows // 2
    central_col = ncols // 2
    return central_row * ncols + central_col

def get_square_index_from_coordinates(
    nrows: int, ncols: int, row: int, col: int
    ) -> int:
    """
    Returns the index of the square in a 2D grid represented as a 1D array.

    Parameters
    ----------
    nrows : int
        The number of rows in the 2D grid.
    ncols : int
        The number of columns in the 2D grid.
    row : int
        The row index of the square.
    col : int
        The column index of the square.

    Returns
    -------
    int
        The index of the square in the 1D array representation of the grid.
    """
    return row * ncols + col

def extract_square(
        dynamic_data: np.ndarray,
        square_size: Union[int, List[int]],
        square_index: Optional[Union[str, int, List[int]]]='central'
    ) -> np.ndarray:
    '''
    Function that extracts a square from the dynamic data. This is useful to
    reduce the size of the data, and also to reduce the number of nodes in the
    graph

    Parameters
    ----------
    dynamic_data: np.ndarray
        tensor of shape (24+, 496, 435, 2)
    square_size: Union[int, List[int]]
        size of the square to extract. If int, then the square will be of size
        (square_size, square_size). If List[int], then the square will be of
        size (square_size[0], square_size[1])
    square_index: Optional[Union[str, int, List[int]]]
        index of the square to extract. If int, then the square will be the
        square_index-th square in the grid. If List[int], then the square will
        be the square with the row index square_index[0] and the column index
        square_index[1]. If 'central', then the square will be the central
        square in the grid
    '''   
    # check if square_size is int or list
    if isinstance(square_size, int):
        square_size = [square_size, square_size]

    # transform shape (24+, 496, 435, 2) -> (496, 435, 2, 24+)
    x = np.moveaxis(dynamic_data, 0, -1)
    
    # calculate the shape of the squares
    nrows = x.shape[0] // square_size[0]
    ncols = x.shape[1] // square_size[1]

    # split on rows
    split1 = np.array_split(
        x,
        range(square_size[0], x.shape[0], square_size[0]),
        axis=0
        )

    # split on columns
    split2 = [
        np.array_split(
            chunk, 
            range(square_size[1], chunk.shape[1], square_size[1]), 
            axis=1
        ) for chunk in split1
    ]

    # flatten the list
    squares = [item for sublist in split2 for item in sublist]

    # Discard any squares that are not of size (p, q)
    squares = [sq for sq in squares if list(sq.shape[:2]) == square_size]
    
    if isinstance(square_index, str):
        if square_index == 'central':
            square_index = get_central_square_index(nrows, ncols)
        else:
            raise ValueError('square_index must be an int or "central"')
    elif isinstance(square_index, list):
        square_index = get_square_index_from_coordinates(
            nrows, ncols, square_index[0], square_index[1]
        )
    
    square = np.array(squares[square_index])

    # move dimensions back to place


    return square


