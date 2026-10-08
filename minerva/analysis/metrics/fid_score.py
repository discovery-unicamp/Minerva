import numpy as np
import scipy


def compute_frechet_distance(ori_representation, gen_representation):
    """Compute Gaussian Fréchet distance from two (samples, features) arrays.

    The sample counts may differ. The arithmetic follows the original
    implementation so existing experiments retain their numerical convention.
    Small negative scores can occur due to floating-point roundoff.
    """
    ori_representation = np.asarray(ori_representation)
    gen_representation = np.asarray(gen_representation)
    for representations in (ori_representation, gen_representation):
        if representations.ndim != 2 or min(representations.shape) < 1:
            raise ValueError("Representations must have shape (n_samples, n_features).")
        if len(representations) < 2:
            raise ValueError("At least two samples are required in each distribution.")
        if not np.isfinite(representations).all():
            raise ValueError("Representations must contain only finite values.")
    if ori_representation.shape[1] != gen_representation.shape[1]:
        raise ValueError(
            "Real and generated representations must have equal dimensions."
        )
    # calculate mean and covariance statistics
    mu1 = ori_representation.mean(axis=0)
    mu2 = gen_representation.mean(axis=0)
    sigma1 = np.atleast_2d(np.cov(ori_representation, rowvar=False))
    sigma2 = np.atleast_2d(np.cov(gen_representation, rowvar=False))
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    if not np.isfinite(fid):
        raise ValueError("Fréchet distance is not finite.")
    return float(fid)


def compute_ts_fid(
    ori_data,
    generated_data,
    *,
    device=0,
    encoder_kwargs=None,
    fit_kwargs=None,
):
    """Train a fresh TS2Vec encoder and compare real and generated series.

    Parameters
    ----------
    ori_data, generated_data : numpy.ndarray
        Arrays of shape (samples, time, channels). Sample counts may differ.
        No axis conversion or normalization is performed here.
    device : str, int or torch.device, optional
        Device passed to TS2Vec. The historical default is device 0.
    encoder_kwargs : dict, optional
        Overrides for TS2Vec hyperparameters, excluding input_dims and device.
        Defaults match the original experiment: batch_size=8, lr=0.001,
        output_dims=320 and max_train_length=3000.
    fit_kwargs : dict, optional
        Arguments to TS2Vec.fit, such as n_iters or n_epochs. By default, use
        TS2Vec's original training budget and verbose=False.

    Returns
    -------
    float
        Fréchet distance in the newly trained TS2Vec representation space.

    Notes
    -----
    Each call trains a new encoder on ori_data only. TS2Vec is an optional
    dependency and is imported only when this function is called.
    """
    for data in (ori_data, generated_data):
        if data.ndim != 3 or min(data.shape) < 1 or len(data) < 2:
            raise ValueError("Series must have shape (n_samples >= 2, time, channels).")
    if ori_data.shape[-1] != generated_data.shape[-1]:
        raise ValueError("Real and generated series must have equal channel counts.")
    options = dict(batch_size=8, lr=0.001, output_dims=320, max_train_length=3000)
    if {"input_dims", "device"}.intersection(encoder_kwargs or {}):
        raise ValueError(
            "Pass device separately; input_dims is inferred from the data."
        )
    options.update(encoder_kwargs or {})
    training_options = {"verbose": False, **(fit_kwargs or {})}
    try:
        from ts2vec.ts2vec import TS2Vec
    except ImportError as exc:
        raise ImportError(
            "TS2Vec evaluation requires a package exposing ts2vec.ts2vec.TS2Vec."
        ) from exc

    model = TS2Vec(input_dims=ori_data.shape[-1], device=device, **options)
    model.fit(ori_data, **training_options)
    ori_representation = model.encode(ori_data, encoding_window="full_series")
    gen_representation = model.encode(generated_data, encoding_window="full_series")
    return compute_frechet_distance(ori_representation, gen_representation)
