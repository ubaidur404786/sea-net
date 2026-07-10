"""
seanet/preprocessing.py - the preprocessing module (how a series becomes "instances").

What this file is for:
    In Multiple Instance Learning a time series is a "bag" of "instances". There are two ways to
    decide what one instance is, and this file switches between them (chosen by config):

      - single_point : one instance = one timestep. A series of length T becomes T instances.
        This is MILLET's original setup and stays exactly as before (this file does nothing to it).

      - sliding_window : one instance = a small window of the series. A series of length T becomes
        N windows, each of `window_size` values, stepped by `window_stride`. The bag is still the
        whole series - only what counts as one instance changes.

    The dataset module (seanet/data.py) only LOADS data; this module re-shapes each series into the
    chosen instance representation. Switching modes needs no code change, only a config change.

How a bag changes shape:
    single_point : bag is (T, 1)      -> T instances, each 1 value.
    sliding_window: bag is (N, W)     -> N instances, each a window of W values.
    The model then transposes to (channels, instances), so W becomes the number of input channels
    and N becomes the sequence length. That is why the model is built with n_in = W in this mode.

What about the metrics:
    - AOPCR works either way (it just removes "instances", which are now windows).
    - NDCG@n needs per-timestep ground-truth labels (only WebTraffic has them), and those cannot be
      lined up with windows, so NDCG is only produced in single_point mode. In sliding_window mode
      the windowed dataset carries no instance labels, so NDCG is simply skipped (like every UCR set).

Related files:
    - seanet/data.py   -> load_dataset() gives us the dataset we re-shape here.
    - seanet/train.py  -> train_one() calls apply_instance_representation() right after loading, and
      builds the model with n_in = window_size when windowing is on.
    - configs/main.yaml -> the "preprocessing" block picks instance_type / window_size / window_stride.
"""
import torch

from millet.data.mil_tsc_dataset import MILTSCDataset


def make_windows(series: torch.Tensor, window_size: int, window_stride: int) -> torch.Tensor:
    """
    Turn one series into a stack of sliding windows.

    Example (window_size=5, window_stride=5): a length-20 series becomes 4 windows:
    [0:5], [5:10], [10:15], [15:20]. Timesteps at the end that do not fill a whole window are
    dropped (this is standard sliding-window behaviour).

    series : one series, shape (T, 1) (or (T,)).
    window_size : how many timesteps are in each window (W).
    window_stride : how far the window moves each step.
    returns : the windows, shape (N, W), where N is the number of windows.
    """
    x = series.reshape(-1)                                    # (T, 1) -> (T,)
    length = x.shape[0]
    if length < window_size:                                 # series shorter than one window -> pad it out
        pad = torch.zeros(window_size - length, dtype=x.dtype)
        x = torch.cat([x, pad])                              # zeros ~= the mean, since the series is normalised
    windows = x.unfold(0, window_size, window_stride)        # (N, W): slide a window of size W with the stride
    return windows.contiguous()


class WindowedDataset(MILTSCDataset):
    """
    A dataset whose bags are sliding windows instead of single timesteps.

    It is built FROM an already-loaded dataset. For each series we first z-normalise it the same way
    MILLET does, then cut it into windows. The windowed bags are stored ready to use, so no transform
    is applied again when you index this dataset. It deliberately carries no instance targets, so the
    NDCG metric is skipped in this mode (see the note at the top of the file).
    """

    def __init__(self, source: MILTSCDataset, window_size: int, window_stride: int):
        """
        source : an already-loaded dataset (from seanet.data.load_dataset).
        window_size : window length W.
        window_stride : step between windows.
        """
        # copy the small bits of state the rest of the pipeline reads
        self.dataset_name = source.dataset_name
        self.split = source.split
        self.window_size = window_size
        self.window_stride = window_stride
        self.n_clz = source.n_clz
        self.targets = source.targets
        self.apply_transform = False                         # bags are already normalised + windowed below

        # normalise each series first (exactly like MILLET), THEN cut it into windows
        self.ts_collection = []
        for i in range(len(source)):
            bag = source.get_bag(i)                          # raw series (T, 1)
            if getattr(source, "apply_transform", True):     # apply the same z-normalisation MILLET uses
                bag = source.apply_bag_transform(bag)
            self.ts_collection.append(make_windows(bag, window_size, window_stride))

    def get_time_series_collection_and_targets(self, split):
        """
        Required by the base class. The data is already built in __init__, so we just return it.
        """
        return self.ts_collection, self.targets


def apply_instance_representation(dataset: MILTSCDataset, instance_type: str,
                                  window_size: int = 5, window_stride: int = 5) -> MILTSCDataset:
    """
    Return the dataset re-shaped into the chosen instance representation.

    dataset : a loaded dataset (from seanet.data.load_dataset).
    instance_type : "single_point" (leave it alone) or "sliding_window" (cut into windows).
    window_size : window length W (only used for sliding_window).
    window_stride : step between windows (only used for sliding_window).
    returns : the same dataset for single_point, or a WindowedDataset for sliding_window.
    raises ValueError : if instance_type is not one of the two options.
    """
    if instance_type == "single_point":
        return dataset                                       # nothing to do - MILLET's original behaviour
    if instance_type == "sliding_window":
        return WindowedDataset(dataset, window_size, window_stride)
    raise ValueError(
        f"Unknown instance_type {instance_type!r} (options: 'single_point', 'sliding_window')."
    )
