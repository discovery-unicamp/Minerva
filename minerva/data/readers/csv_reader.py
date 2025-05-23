from typing import List, Optional, Tuple, Union

import pandas as pd

from minerva.utils.typing import PathLike
from minerva.data.readers.tabular_reader import TabularReader


class CSVReader(TabularReader):
    def __init__(
        self,
        path: Union[PathLike, pd.DataFrame],
        columns_to_select: Union[str, List[str]],
        cast_to: Optional[str] = None,
        data_shape: Optional[Tuple[int, ...]] = None,
        reader_kwargs: Optional[dict] = None,
    ):
        reader_kwargs = reader_kwargs or {}
        if isinstance(path, pd.DataFrame):
            df = path
        else:
            df = pd.read_csv(path, **reader_kwargs)

        super().__init__(df, columns_to_select, cast_to, data_shape)
