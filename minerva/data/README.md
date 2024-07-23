# Data

## Readers

| **Reader**         | **Data Unit**                                                                      |       **Order**       |     **Class**      | **Observations**                                                                                                                    |
| :----------------- | ---------------------------------------------------------------------------------- | :-------------------: | :----------------: | ----------------------------------------------------------------------------------------------------------------------------------- |
| PNGReader          | Each unit of data is a image file (PNG) inside the root folder                     | Lexicographical order |     PNGReader      | File extensions: .png                                                                                                               |
| TIFFReader         | Each unit of data is a image file (TIFF) inside the root folder                    | Lexicographical order |     TiffReader     | File extensions: .tif and .tiff                                                                                                     |
| TabularReader      | Each unit of data is the i-th row in a dataframe, with columns filtered            |    Dataframe rows     |   TabularReader    | Support pandas dataframe                                                                                                            |
| CSVReader          | Each unit of data is the i-th row in a CSV file, with columns filtered             |       CSV Rowd        |     CSVReader      | If data frame is already open, use TabularReader instead. This class will open and load the CSV file and pass it to a TabularReader |
| PatchedArrayReader | Each unit of data is a sub matrix of specified shape inside an n-dimensional array |    Dimension order    | PatchedArrayReader | Supports any data with ndarray protocol (tensor, xarray, zarr)                                                                      |
| PatchedZarrReader  | Each unit of data is a sub matrix of specified shape inside an Zarr Array          |    Dimension order    |  ZarrArrayReader   | Open zarr file in lazy mode and pass it to PatchedArrayReader                                                                       |
