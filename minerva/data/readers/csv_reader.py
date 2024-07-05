from typing import List, Optional, Tuple, Union

import pandas as pd

from minerva.data.readers.tabular_reader import TabularReader


class CSVReader(TabularReader):
    def __init__(
        self,
        path: str,
        columns_to_select: Union[str, List[str]],
        cast_to: Optional[str] = None,
        data_shape: Optional[Tuple[int, ...]] = None,
    ):
        df = pd.read_csv(path)
        super().__init__(df, columns_to_select, cast_to, data_shape)
