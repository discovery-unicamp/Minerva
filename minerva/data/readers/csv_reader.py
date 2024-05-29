from typing import Union

import pandas as pd
from minerva.data.readers.tabular_reader import TabularReader

class CSVReader(TabularReader):
    def __init__(
        self,
        path: str,
        columns_to_select: Union[str, list[str]],
        cast_to: str = None,
        data_shape: tuple[int, ...] = None,
    ):
        df = pd.read_csv(path)
        super().__init__(df, columns_to_select, cast_to, data_shape)
        
        
