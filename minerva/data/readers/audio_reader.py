from typing import Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from collections import OrderedDict
from minerva.data.readers.reader import _Reader
import math, torch, julius, os, tqdm, json
import torch.nn.functional as F
import torchaudio as ta

# Based off the class Wavset
# avaliable at https://github.com/facebookresearch/demucs/blob/main/demucs/wav.py


def _compute_metadata(
    track_dir: Path,
    target_stem: str = "mixture",
    stems: Optional[List[str]] = None,
    extension: str = "wav",
) -> Dict:
    """Computes the mean and standard deviation for a target stem in a given directory
    of the dataset. The directory must have the structure

    ```
    /track_dir/
    ├── mixture.wav
    ├── stem1.wav
    ├── stem2.wav
    └── ...
    ```

    Parameters
    ----------
    track_dir : Path
        Path to the directory.
    target_stem : str
        The name of the target stem from which to calculate the metadata. Defaults to
        `"mixture"`
    stems : List[str], optional
        If set, will ensure all stems have the same length and sample rate
    extension : str
        The extension for the audio files. Defaults to `"wav"`

    Returns
    -------
    dict
        A dictionary containing the keys `"length"`, `"mean"`, `"std"` and `"sample_rate"`

    Raises
    ------
        ValueError
            If the informed stems have different lengths or sample rates

        RuntimeError
            If torchaudio is unable to load the audio files.
    """

    track_length = None
    track_sample_rate = None
    stems = stems or []

    # Assuring all files have the same length and sample rate
    for stem in stems:
        stem_path = track_dir / f"{stem}.{extension}"

        try:
            info = ta.info(str(stem_path))
        except RuntimeError:
            print(stem_path)
            raise

        if track_length is None:
            track_length = info.num_frames
            track_sample_rate = info.sample_rate
        elif track_length != info.num_frames:
            raise ValueError(
                f"Invalid length for file {stem_path}: "
                f"expecting {track_length} but got {info.num_frames}"
            )
        elif track_sample_rate != info.sample_rate:
            raise ValueError(
                f"Invalid sample rate for file {stem_path}: "
                f"expecting {track_sample_rate} but got {info.sample_rate}"
            )

    # Computing statistics
    mixture_path = str(track_dir / f"{target_stem}.{extension}")
    try:
        data, _ = ta.load(mixture_path)
        info = ta.info(mixture_path)
    except RuntimeError:
        print(mixture_path)
        raise

    data = data.mean(0)  # Removing channels
    mean = data.mean().item()
    std = data.std().item()

    return {
        "length": info.num_frames,
        "mean": mean,
        "std": std,
        "sample_rate": info.sample_rate,
    }


def build_metadata(
    path: Union[str, Path],
    target_stem: str = "mixture",
    stems: Optional[List[str]] = None,
    extension: str = "wav",
) -> Dict:
    """Builds the metadata for an audio dataset, containing the name, length, mean,
    standard deviation and sample rate for each directory in the dataset. The directory
    must have the structure

    ```
    /path/
    ├── /audio1/
    │   ├── mixture.wav
    │   ├── stem1.wav
    │   ├── stem2.wav
    │   └── ...
    ├── /audio2/
    │   ├── mixture.wav
    │   ├── stem1.wav
    │   ├── stem2.wav
    │   └── ...
    └── ...
    ```

    Parameters
    ----------
    path : Path
        Path to the dataset.
    target_stem : str
        The name of the target stem from which to calculate the metadata. Defaults to
        `"mixture"`
    stems : List[str], optional
        If set, will ensure all stems have the same length and sample rate
    extension : str
        The extension for the audio files. Defaults to `"wav"`

    Returns
    -------
    dict
        A dictionary where the keys are the directory names and the values are
        dictionaries containing the keys `"length"`, `"mean"`, `"std"` and `"sample_rate"`
    """

    metadata = {}
    stems = stems or []
    path = Path(path)
    pendings = []

    with ThreadPoolExecutor(8) as pool:
        for root, folders, _ in os.walk(path, followlinks=True):
            root = Path(root)

            # Skip hidden dirs and dirs that contain subdirs
            if root.name.startswith(".") or folders or root == path:
                continue

            name = str(root.relative_to(path))
            pendings.append(
                (
                    name,
                    pool.submit(_compute_metadata, root, target_stem, stems, extension),
                )
            )

    for name, pending in tqdm.tqdm(pendings):
        metadata[name] = pending.result()

    return metadata


class AudioReader(_Reader):
    """A reader for audio datasets. May automatically extracts exerpts from long audio
    files. It assumes all related audio files ("stems") are bundled together in the same
    directory and consistently named across the dataset. Therefore the data must have
    the structure.
    
    ```
    /dataset/
    ├── /song1/
    │   ├── mixture.wav
    │   ├── vocals.wav
    │   └── accompaniment.wav
    ├── /song2/
    │   ├── mixture.wav
    │   ├── vocals.wav
    │   └── accompaniment.wav
    └── ...
    ```

    Supports the same extensions as torchaudio, which may vary depending on installation.
    
    """

    def __init__(
        self,
        root: Union[str, Path],
        metadata: Union[Dict, str, Path],
        stems: List[str],
        segment: Optional[float] = None,
        stride: Optional[float] = None,
        sample_rate: int = 44_100,
        channels: int = 2,
        normalize: bool = True,
        extension: str = "wav",
    ):
        """Loads audio files from a directory. See class documentation for the required
        directory structure.
        
        Parameters
        ----------
        root : Union[str, Path]
            Path to the dataset's root directory.
        metadata : Union[Dict, str, Path]
            A dictionary containing the metadata generated by audio_reader.build_metadata
            or a path to a json file containing it.
        stems : List[str]
            A list containing at least one stem determining which files are read.
        segment : float, optional
            The length in seconds for audio samples to be extracted. If not set, reads
            the whole file.
        stride : float, optional
            The stride in seconds between the beginning of a sample and the beginning of
            the next one. If not set, defaults to the segment length (i.e. no overlap
            between samples).
        sample_rate : int
            The sample rate for all samples. If the file a sample is read from has a
            different sample rate, the sample will be converted on the fly. Defaults to
            44,100.
        channels : int
            The number of audio channels for all samples. If set to 1 and file is
            stereo, audio is converted to mono on the fly. If set to n and file is mono,
            the mono channel is copied n times. Otherwise, selects the n first channels.
            Defaults to 2.
        normalize : bool
            Whether or not to normalize audio (mean 0, standard deviation 1) before
            yielding reader entry.
        extension : str
            The extension of the audio files. Defaults to `"wav"`.
        """

        self.root = Path(root)
        self.segment = segment
        self.stride = stride or segment
        self.stems = stems
        self.sample_rate = sample_rate
        self.channels = channels
        self.normalize = normalize
        self.extension = extension

        if isinstance(metadata, dict):
            self.metadata = OrderedDict(metadata)
        else:
            self.metadata = self._load_metadata(metadata)

        self.num_samples = []

        for song_metadata in self.metadata.values():

            track_duration = song_metadata["length"] / song_metadata["sample_rate"]

            if segment is None or track_duration < segment:
                self.num_samples.append(1)
            else:
                examples = math.ceil((track_duration - self.segment) / self.stride) + 1
                self.num_samples.append(examples)

    def __len__(self) -> int:
        """Gets the number of samples in the dataset.
        
        Returns
        -------
        int
            The number of samples in the dataset.
        """
        return sum(self.num_samples)

    def __getitem__(self, index: int) -> torch.Tensor:
        """ Gets the audio sample at the specified index.
        
        Parameters
        ----------
        index : int
            Index of the audio sample.
        
        Returns
        -------
        torch.Tensor
            An (S, C, L) tensor where S is the number of stems, C is the number of
            channels and L is the length of audio in frames.
        """
        for song_name, n_examples in zip(self.metadata, self.num_samples):

            if index >= n_examples:
                index -= n_examples
                continue

            song_metadata = self.metadata[song_name]
            offset = int(song_metadata["sample_rate"] * self.stride * index)
            num_frames = math.ceil(song_metadata["sample_rate"] * self.segment)

            stem_excerpts = []
            for stem in self.stems:
                path = self.root / song_name / f"{stem}.{self.extension}"
                data, _ = ta.load(str(path), offset, num_frames)
                data = self._convert_channels(data, self.channels)
                stem_excerpts.append(data)

            stem_excerpts = torch.stack(stem_excerpts)
            stem_excerpts = julius.resample_frac(
                stem_excerpts, song_metadata["sample_rate"], self.sample_rate
            )

            if self.normalize:
                stem_excerpts -= song_metadata["mean"]
                stem_excerpts /= song_metadata["std"]

            length = int(self.segment * self.sample_rate)
            stem_excerpts = stem_excerpts[..., :length]
            stem_excerpts = F.pad(stem_excerpts, (0, length - stem_excerpts.shape[-1]))
            return stem_excerpts

    def _load_metadata(self, path: Union[str, Path]) -> Dict:
        """Loads the dataset metadata stored in a json file.
        
        Parameters
        ----------
        path : Union[str, Path]
            Path to the json file.
        
        Returns
        -------
        dict
            The dataset metadata.
        
        Raises
        ------
        ValueError
            If file does not exist or does not contain valid json.
        """
        path = Path(path)

        if not path.is_file():
            raise ValueError(
                f"{path.absolute()} does not exist or isn't a file"
            )

        # Fetch json
        with open(path) as f:
            try:
                metadata = json.load(f)
            except json.JSONDecodeError:
                raise ValueError(
                    f"{path.absolute()} could not be decoded. "
                    f"Make sure this metadata file is a dictionary generated by "
                    f"audio_reader.build_metadata"
                )

        return metadata

    def _convert_channels(self, audio: torch.Tensor, channels: int) -> torch.Tensor:
        """Converts audio to the specified number of channels.
        
        Parameters
        ----------
        audio : torch.Tensor
            A tensor representing audio. Shape must be (*, C, L) where C are the
            channels and L is the length in frames.
        channels : int
            The number of channels of the output.
        
        Returns
        -------
        torch.Tensor
            The converted audio.
        
        Raises
        ------
        ValueError
            If audio file has fewer channels than requested and is not mono.
        """
        *shape, src_channels, length = audio.shape
        if src_channels == channels:
            pass
        elif channels == 1:
            # Case 1:
            # The caller asked 1-channel audio, but the stream have multiple
            # channels, downmix all channels.
            audio = audio.mean(dim=-2, keepdim=True)
        elif src_channels == 1:
            # Case 2:
            # The caller asked for multiple channels, but the input file have
            # one single channel, replicate the audio over all channels.
            audio = audio.expand(*shape, channels, length)
        elif src_channels >= channels:
            # Case 3:
            # The caller asked for multiple channels, and the input file have
            # more channels than requested. In that case return the first channels.
            audio = audio[..., :channels, :]
        else:
            # Case 4: What is a reasonable choice here?
            raise ValueError(
                "The audio file has fewer channels than requested but is not mono."
            )
        return audio
