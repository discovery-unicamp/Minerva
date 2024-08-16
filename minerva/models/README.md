
# Models

The models module is a collection of models that can be used for various tasks. I has two main submodules: `nets` and `ssl`. The `nets` submodule contains architectures for neural networks that can be trained in a supervised manner. The `ssl` submodule contains implementations of semi-supervised learning techniques, usually requiring a architecture from the `nets` submodule to run.

Both `nets` and `ssl` submodules are further divided into areas of use (e.g. `image` and `time_series`). This division is made to make it easier to find the right model for the right task. It's not uncommon for a model to be used in more than one area of use, in that case it will be in the main area of use in it's original paper.

In the case of a architecture or semi-supervised learning technique been agnostic to the type of data, it will be in the root of the submodule.

## Nets

These are the models implemented in the `nets` submodule:

| **Model**                                                                  | **Authors**                      |    **Task**    |       **Type**        |       **Input Shape**       |   **Python Class**    | **Observations**                                                                                                            |
| -------------------------------------------------------------------------- | -------------------------------- | :------------: | :-------------------: | :-------------------------: | :-------------------: | --------------------------------------------------------------------------------------------------------------------------- |
| [Deep Conv LSTM](https://www.mdpi.com/1424-8220/16/1/115)                  | Ordóñez and Roggen               | Classification |    2D Conv + LSTM     |     (C,&#160;S,&#160;T)     |     DeepConvLSTM      | --                                                                                                                          |
| [Simple 1D Convolutional Network](https://www.mdpi.com/1424-8220/16/1/115) | Ordóñez and Roggen               | Classification |        1D Conv        |         (S,&#160;T)         |  Simple1DConvNetwork  | 1D Variant of "Baseline CNN", used by Ordóñez and Roggen,  with dropout layers included.                                    |
| [Simple 2D Convolutional Network](https://www.mdpi.com/1424-8220/16/1/115) | Ordóñez and Roggen               | Classification |        2D Conv        |     (C,&#160;S,&#160;T)     |  Simple2DConvNetwork  | 2D Variant of "Baseline CNN", used by Ordóñez and Roggen,  with dropout layers included.                                    |
| [CNN Ha *et al.* 1D](https://ieeexplore.ieee.org/document/7379657)         | Ha, Yun and Choi                 | Classification |        1D Conv        |         (S,&#160;T)         |     CNN_HaEtAl_1D     | 1D proposed variant.                                                                                                        |
| [CNN Ha *et al.* 2D](https://ieeexplore.ieee.org/document/7379657)         | Ha, Yun and Choi                 | Classification |        2D Conv        |     (C,&#160;S,&#160;T)     |     CNN_HaEtAl_2D     | 2D proposed variant.                                                                                                        |
| [CNN PF](https://ieeexplore.ieee.org/document/7727224)                     | Ha and Choi                      | Classification |        2D Conv        |     (C,&#160;S,&#160;T)     |       CNN_PF_2D       | Partial weight sharing in first convolutional layer and  full weight sharing in second convolutional layer.                 |
| [CNN PPF](https://ieeexplore.ieee.org/document/7727224)                    | Ha and Choi                      | Classification |        2D Conv        |     (C,&#160;S,&#160;T)     |      CNN_PFF_2D       | Partial and full weight sharing in first convolutional layer  and full weight sharing in second convolutional layer.        |
| [IMU Transformer](https://ieeexplore.ieee.org/document/9393889)            | Shavit and Klein                 | Classification | 1D Conv + Transformer |         (S,&#160;T)         | IMUTransformerEncoder | --                                                                                                                          |
| [IMU CNN](https://ieeexplore.ieee.org/document/9393889)                    | Shavit and Klein                 | Classification |        1D Conv        |         (S,&#160;T)         |        IMUCNN         | Baseline CNN for IMUTransnformer work.                                                                                      |
| [Inception Time](https://doi.org/10.1007/s10618-020-00710-y)               | Fawaz *et al.*                   | Classification |        1D Conv        |         (S,&#160;T)         |     InceptionTime     | --                                                                                                                          |
| [1D ResNet](https://www.mdpi.com/1424-8220/22/8/3094)                      | Mekruksavanich and Jitpattanakul | Classification |        1D Conv        |         (S,&#160;T)         |      ResNet1D_8       | Baseline resnet from paper. Uses ELU and 8 residual blocks                                                                  |
| [1D ResNet SE 8](https://www.mdpi.com/1424-8220/22/8/3094)                 | Mekruksavanich and Jitpattanakul | Classification |        1D Conv        |         (S,&#160;T)         |     ResNetSE1D_8      | ResNet with Squeeze and Excitation. Uses ELU and 8 residual  blocks                                                         |
| [1D ResNet SE 5](https://ieeexplore.ieee.org/document/9771436)             | Mekruksavanich *et al.*          | Classification |        1D Conv        |         (S,&#160;T)         |     ResNetSE1D_5      | ResNet with Squeeze and Excitation. Uses ReLU and 8 residual  blocks                                                        |
| [MCNN](https://ieeexplore.ieee.org/document/8975649)                       | Sikder *et al.*                  | Classification |        2D Conv        | (2,&#160;C,&#160;S,&#160;T) |  MultiChannelCNN_HAR  | First dimension is FFT data and second is Welch Power Density periodgram data. Must adapt dataset to return data like this. |

## SSL Models

These are the models implemented in the `ssl` submodule:

| **Model**                               | **Authors**     | **Task** | **Type** | **Input Shape** |     **Python Class**     | **Observations** |
| --------------------------------------- | --------------- | -------- | -------- | :-------------: | :----------------------: | ---------------- |
| [LFR](https://arxiv.org/abs/2310.07756) | Yi Sui *et al.* | Any      | Any      |       Any       | LearnFromRandomnessModel |                  |
|[TF-C](https://arxiv.org/abs/2206.08496) | Zhang et al.    | Classification 	| 1D Conv	| (C, S, T)     	| TFC_Model 	| Default backbone is the convolutional with a MLP as prediction head |