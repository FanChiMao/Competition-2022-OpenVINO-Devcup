from typing import Optional
import itertools
import numpy as np

def _norm(x):
    return np.abs(x[..., 0]) ** 2 + np.abs(x[..., 1]) ** 2

def _mul_add(a, b, out=None):
    """Element-wise multiplication of two complex Tensors described
    through their real and imaginary parts.
    The result is added to the `out` tensor"""

    # check `out` and allocate it if needed
    target_shape = tuple([max(sa, sb) for (sa, sb) in zip(a.shape, b.shape)])
    if out is None or out.shape != target_shape:
        out = np.zeros(target_shape, dtype=a.dtype)
    if out is a:
        real_a = a[..., 0]
        out[..., 0] = out[..., 0] + (real_a * b[..., 0] - a[..., 1] * b[..., 1])
        out[..., 1] = out[..., 1] + (real_a * b[..., 1] + a[..., 1] * b[..., 0])
    else:
        out[..., 0] = out[..., 0] + (a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1])
        out[..., 1] = out[..., 1] + (a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0])
    return out

def _conj(z, out=None):
    """Element-wise complex conjugate of a Tensor with complex entries
    described through their real and imaginary parts.
    can work in place in case out is z"""
    if out is None or out.shape != z.shape:
        out = np.zeros_like(z)
    out[..., 0] = z[..., 0]
    out[..., 1] = -z[..., 1]
    return out

def _covariance(y_j):
    """
    Compute the empirical covariance for a source.
    Args:
        y_j (Tensor): complex stft of the source.
            [shape=(nb_frames, nb_bins, nb_channels, 2)].
    Returns:
        Cj (Tensor): [shape=(nb_frames, nb_bins, nb_channels, nb_channels, 2)]
            just y_j * conj(y_j.T): empirical covariance for each TF bin.
    """
    (nb_frames, nb_bins, nb_channels) = y_j.shape[:-1]
    Cj = np.zeros(
        (nb_frames, nb_bins, nb_channels, nb_channels, 2),
        dtype=y_j.dtype
    )
    indices = [element for element in itertools.product(np.arange(nb_channels), np.arange(nb_channels))]
    for index in indices:
        Cj[:, :, index[0], index[1], :] = _mul_add(
            y_j[:, :, index[0], :],
            _conj(y_j[:, :, index[1], :]),
            Cj[:, :, index[0], index[1], :],
        )
    return Cj

def _mul(a, b, out=None):
    """Element-wise multiplication of two complex Tensors described
    through their real and imaginary parts
    can work in place in case out is a only"""
    target_shape = tuple([max(sa, sb) for (sa, sb) in zip(a.shape, b.shape)])
    if out is None or out.shape != target_shape:
        out = np.zeros(target_shape, dtype=a.dtype)
    if out is a:
        real_a = a[..., 0]
        out[..., 0] = real_a * b[..., 0] - a[..., 1] * b[..., 1]
        out[..., 1] = real_a * b[..., 1] + a[..., 1] * b[..., 0]
    else:
        out[..., 0] = a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1]
        out[..., 1] = a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0]
    return out

def _inv(z, out=None):
    """Element-wise multiplicative inverse of a Tensor with complex
    entries described through their real and imaginary parts.
    can work in place in case out is z"""
    ez = _norm(z)
    if out is None or out.shape != z.shape:
        out = np.zeros_like(z)
    out[..., 0] = z[..., 0] / ez
    out[..., 1] = -z[..., 1] / ez
    return out

def _invert(M, out=None):
    """
    Invert 1x1 or 2x2 matrices
    Will generate errors if the matrices are singular: user must handle this
    through his own regularization schemes.
    Args:
        M (Tensor): [shape=(..., nb_channels, nb_channels, 2)]
            matrices to invert: must be square along dimensions -3 and -2
    Returns:
        invM (Tensor): [shape=M.shape]
            inverses of M
    """
    nb_channels = M.shape[-2]

    if out is None or out.shape != M.shape:
        out = np.empty_like(M)

    if nb_channels == 1:
        # scalar case
        out = _inv(M, out)
    elif nb_channels == 2:
        # two channels case: analytical expression

        # first compute the determinent
        det = _mul(M[..., 0, 0, :], M[..., 1, 1, :])
        det = det - _mul(M[..., 0, 1, :], M[..., 1, 0, :])
        # invert it
        invDet = _inv(det)

        # then fill out the matrix with the inverse
        out[..., 0, 0, :] = _mul(invDet, M[..., 1, 1, :], out[..., 0, 0, :])
        out[..., 1, 0, :] = _mul(-invDet, M[..., 1, 0, :], out[..., 1, 0, :])
        out[..., 0, 1, :] = _mul(-invDet, M[..., 0, 1, :], out[..., 0, 1, :])
        out[..., 1, 1, :] = _mul(invDet, M[..., 0, 0, :], out[..., 1, 1, :])
    else:
        raise Exception("Only 2 channels are supported for the torch version.")
    return out

    
def expectation_maximization(
    y,
    x,
    iterations: int = 2,
    eps: float = 1e-10,
    batch_size: int = 200,
):
    r"""Expectation maximization algorithm, for refining source separation
    estimates.
    This algorithm allows to make source separation results better by
    enforcing multichannel consistency for the estimates. This usually means
    a better perceptual quality in terms of spatial artifacts.
    The implementation follows the details presented in [1]_, taking
    inspiration from the original EM algorithm proposed in [2]_ and its
    weighted refinement proposed in [3]_, [4]_.
    It works by iteratively:
     * Re-estimate source parameters (power spectral densities and spatial
       covariance matrices) through :func:`get_local_gaussian_model`.
     * Separate again the mixture with the new parameters by first computing
       the new modelled mixture covariance matrices with :func:`get_mix_model`,
       prepare the Wiener filters through :func:`wiener_gain` and apply them
       with :func:`apply_filter``.
    References
    ----------
    .. [1] S. Uhlich and M. Porcu and F. Giron and M. Enenkl and T. Kemp and
        N. Takahashi and Y. Mitsufuji, "Improving music source separation based
        on deep neural networks through data augmentation and network
        blending." 2017 IEEE International Conference on Acoustics, Speech
        and Signal Processing (ICASSP). IEEE, 2017.
    .. [2] N.Q. Duong and E. Vincent and R.Gribonval. "Under-determined
        reverberant audio source separation using a full-rank spatial
        covariance models." IEEE Transactions on Audio, Speech, and Language
        Processing 18.7 (2010): 1830-1840.
    .. [3] A. Nugraha and A. Liutkus and E. Vincent. "Multichannel audio source
        separation with deep neural networks." IEEE/ACM Transactions on Audio,
        Speech, and Language Processing 24.9 (2016): 1652-1664.
    .. [4] A. Nugraha and A. Liutkus and E. Vincent. "Multichannel music
        separation with deep neural networks." 2016 24th European Signal
        Processing Conference (EUSIPCO). IEEE, 2016.
    .. [5] A. Liutkus and R. Badeau and G. Richard "Kernel additive models for
        source separation." IEEE Transactions on Signal Processing
        62.16 (2014): 4298-4310.
    Args:
        y (Tensor): [shape=(nb_frames, nb_bins, nb_channels, 2, nb_sources)]
            initial estimates for the sources
        x (Tensor): [shape=(nb_frames, nb_bins, nb_channels, 2)]
            complex STFT of the mixture signal
        iterations (int): [scalar]
            number of iterations for the EM algorithm.
        eps (float or None): [scalar]
            The epsilon value to use for regularization and filters.
    Returns:
        y (Tensor): [shape=(nb_frames, nb_bins, nb_channels, 2, nb_sources)]
            estimated sources after iterations
        v (Tensor): [shape=(nb_frames, nb_bins, nb_sources)]
            estimated power spectral densities
        R (Tensor): [shape=(nb_bins, nb_channels, nb_channels, 2, nb_sources)]
            estimated spatial covariance matrices
    Notes:
        * You need an initial estimate for the sources to apply this
          algorithm. This is precisely what the :func:`wiener` function does.
        * This algorithm *is not* an implementation of the "exact" EM
          proposed in [1]_. In particular, it does compute the posterior
          covariance matrices the same (exact) way. Instead, it uses the
          simplified approximate scheme initially proposed in [5]_ and further
          refined in [3]_, [4]_, that boils down to just take the empirical
          covariance of the recent source estimates, followed by a weighted
          average for the update of the spatial covariance matrix. It has been
          empirically demonstrated that this simplified algorithm is more
          robust for music separation.
    Warning:
        It is *very* important to make sure `x.dtype` is `torch.float64`
        if you want double precision, because this function will **not**
        do such conversion for you from `torch.complex32`, in case you want the
        smaller RAM usage on purpose.
        It is usually always better in terms of quality to have double
        precision, by e.g. calling :func:`expectation_maximization`
        with ``x.to(torch.float64)``.
    """
    # dimensions
    (nb_frames, nb_bins, nb_channels) = x.shape[:-1]
    nb_sources = y.shape[-1]

    regularization = np.concatenate(
        (
            np.eye(nb_channels, dtype=x.dtype)[..., None],
            np.zeros((nb_channels, nb_channels, 1), dtype=x.dtype),
        ),
        axis=2,
    )
    regularization = np.sqrt(np.asarray(eps)) * (
        #regularization[None, None, ...].expand((-1, nb_bins, -1, -1, -1))
        np.tile(regularization[None, None, ...], ((1, nb_bins, 1, 1, 1)))
    )

    # allocate the spatial covariance matrices
    R = [
        np.zeros((nb_bins, nb_channels, nb_channels, 2), dtype=x.dtype)
        for j in range(nb_sources)
    ]
    weight = np.zeros((nb_bins,), dtype=x.dtype)

    v = np.zeros((nb_frames, nb_bins, nb_sources), dtype=x.dtype)
    for it in range(iterations):
        # constructing the mixture covariance matrix. Doing it with a loop
        # to avoid storing anytime in RAM the whole 6D tensor

        # update the PSD as the average spectrogram over channels
        v = np.mean(np.abs(y[..., 0, :]) ** 2 + np.abs(y[..., 1, :]) ** 2, axis=-2)

        # update spatial covariance matrices (weighted update)
        for j in range(nb_sources):
            R[j] = np.array(0.0)
            weight = np.array(eps)
            pos: int = 0
            batch_size = batch_size if batch_size else nb_frames
            while pos < nb_frames:
                t = np.arange(pos, min(nb_frames, pos + batch_size))
                pos = int(t[-1]) + 1

                R[j] = R[j] + np.sum(_covariance(y[t, ..., j]), axis=0)
                weight = weight + np.sum(v[t, ..., j], axis=0)
            R[j] = R[j] / weight[..., None, None, None]
            weight = np.zeros_like(weight)

        pos = 0
        while pos < nb_frames:
            t = np.arange(pos, min(nb_frames, pos + batch_size))
            pos = int(t[-1]) + 1

            y[t, ...] = np.array(0.0, dtype=x.dtype)

            # compute mix covariance matrix
            Cxx = regularization
            for j in range(nb_sources):
                Cxx = Cxx + (v[t, ..., j, None, None, None] * R[j][None, ...].copy())

            # invert it
            inv_Cxx = _invert(Cxx)

            # separate the sources
            for j in range(nb_sources):

                # create a wiener gain for this source
                gain = np.zeros_like(inv_Cxx)

                # computes multichannel Wiener gain as v_j R_j inv_Cxx
                indices = [element for element in itertools.product(np.arange(nb_channels), np.arange(nb_channels), np.arange(nb_channels))]
                for index in indices:
                    gain[:, :, index[0], index[1], :] = _mul_add(
                        R[j][None, :, index[0], index[2], :].copy(),
                        inv_Cxx[:, :, index[2], index[1], :],
                        gain[:, :, index[0], index[1], :],
                    )
                gain = gain * v[t, ..., None, None, None, j]

                # apply it to the mixture
                for i in range(nb_channels):
                    y[t, ..., j] = _mul_add(gain[..., i, :], x[t, ..., i, None, :], y[t, ..., j])

    return y, v, R

def wiener(targets_spectrograms, mix_stft, iterations, scale_factor=10.0, eps=1e-10):
    # otherwise, we just multiply the targets spectrograms with mix phase
    # we tacitly assume that we have magnitude estimates.
    angle = np.arctan2(mix_stft.imag, mix_stft.real)
    angle = np.expand_dims(angle, axis=-1)
    
    nb_sources = targets_spectrograms.shape[-1]
    
    mix_stft = np.stack([mix_stft.real, mix_stft.imag], axis=-1)
    mix_stft = mix_stft.astype("float32")
    y = np.zeros(mix_stft.shape + (nb_sources,))
    y[..., 0, :] = targets_spectrograms * np.cos(angle)
    y[..., 1, :] = targets_spectrograms * np.sin(angle)
    y = y.astype("float32")
        
    max_abs = np.max((1.0, np.sqrt(_norm(mix_stft)).max()/scale_factor))
    
    mix_stft /= max_abs
    y /= max_abs
    
    y = expectation_maximization(y, mix_stft, iterations, eps=eps)[0]
    y *= max_abs
    
    return y