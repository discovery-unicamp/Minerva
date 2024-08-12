import re
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from minerva.data.readers.reader import _Reader


class TabularReader(_Reader):
    def __init__(
        self,
        df: pd.DataFrame,
        columns_to_select: Union[str, List[str]],
        cast_to: Optional[str] = None,
        data_shape: Optional[Tuple[int, ...]] = None,
    ):
        """Reader to select columns from a DataFrame and return them as a NumPy
        array. The DataFrame is indexed by the row number. Each row of the
        DataFrame is considered as a sample. Thus, the __getitem__ method will
        return the columns of the DataFrame at the specified index as a NumPy
        array.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to select the columns from. The DataFrame should have
            the columns that are specified in the `columns_to_select` parameter.
        columns_to_select : Union[str, list[str]]
            A string or a list of strings used to select the columns from the DataFrame.
            The string can be a regular expression pattern or a column name. The columns
            that match the pattern will be selected.
        cast_to : str, optional
            Cast the selected columns to the specified data type. If None, the
            data type of the columns will not be changed. (default is None)
        data_shape : tuple[int, ...], optional
            The shape of the data to be returned. If None, the data will be
            returned as a 1D array. If provided, the data will be reshaped to
            the specified shape. (default is None)
        """
        self.df = df
        self.columns_to_select = columns_to_select
        self.cast_to = cast_to
        self.data_shape = data_shape

        if isinstance(self.columns_to_select, str):
            self.columns_to_select = [self.columns_to_select]

    def __getitem__(self, index: int) -> np.ndarray:
        """Return the columns of the DataFrame at the specified row index as a NumPy
        array. The columns are selected based on the `self.columns_to_select`.

        Parameters
        ----------
        index : int
            The row index to select the columns from the DataFrame.

        Returns
        -------
        np.ndarray
            The selected columns from the row as a NumPy array.
        """
        columns = list(self.df.columns)

        # Filter valid columns based on columns_to_select list
        valid_columns = []
        for pattern in self.columns_to_select:
            valid_columns.extend([col for col in columns if re.match(pattern, col)])

        # Select the elements and return
        row = self.df.iloc[index][valid_columns]
        row = row.to_numpy()

        if self.cast_to is not None:
            row = row.astype(self.cast_to)

        if self.data_shape is not None:
            row = row.reshape(self.data_shape)

        return row

    def __len__(self) -> int:
        """Return the number of samples in the DataFrame. The number of samples
        is equal to the number of rows in the DataFrame.

        Returns
        -------
        int
            The number of samples in the DataFrame.
        """
        return len(self.df)


# def main():
#     df = pd.DataFrame({
#         "accel-x-0": np.array(range(10)),
#         "accel-x-1": np.array(range(10)) + 10,
#         "accel-x-2": np.array(range(10)) + 100,
#         "accel-x-3": np.array(range(10)) + 1000,

#         "accel-y-0": np.array(range(10)),
#         "accel-y-1": np.array(range(10)) * 2,
#         "accel-y-2": np.array(range(10)) * 3,
#         "accel-y-3": np.array(range(10)) * 4,

#         "gyro-x-0": np.array(range(10)) - 10,
#         "gyro-x-1": np.array(range(10)) - 20,
#         "gyro-x-2": np.array(range(10)) - 30,
#         "gyro-x-3": np.array(range(10)) - 40,
#     })

#     reader = TabularReader(df, ["accel-x-*", "gyro-x-*"])
#     print(len(reader))
#     print(reader[1])

#     reader = TabularReader(df, ["accel-*", "gyro-x-*"])
#     print(len(reader))
#     print(reader[2])


#     reader = TabularReader(df, ["accel-x-1", "gyro-x-0", "gyro-x-1", "accel-y-*"])
#     print(len(reader))
#     print(reader[3])


# if __name__ == "__main__":
#     main()
