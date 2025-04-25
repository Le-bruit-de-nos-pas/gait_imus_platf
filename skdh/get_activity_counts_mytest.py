from numpy import array, repeat, abs, minimum, floor, float64, nonzero, interp
from scipy.signal import lfilter_zi, lfilter

from sdk.utility.internal import apply_resample
from sdk.utility import moving_mean

__all__ = ["get_activity_counts"]

input_coef = array(
    [
        -0.009341062898525,
        -0.025470289659360,
        -0.004235264826105,
        0.044152415456420,
        0.036493718347760,
        -0.011893961934740,
        -0.022917390623150,
        -0.006788163862310,
        0.000000000000000,
    ],
    dtype=float64
)

output_coef = array(
    [
        1.00000000000000000000,
        -3.63367395910957000000,
        5.03689812757486000000,
        -3.09612247819666000000,
        0.50620507633883000000,
        0.32421701566682000000,
        -0.15685485875559000000,
        0.01949130205890000000,
        0.00000000000000000000,
    ],
    dtype=float64,
)



def moving_mean(a, w_len, skip, trim=True, axis=1):
    if w_len <=0 or skip <= 0:
        raise ValueError("`wlen` and `skip` cannot be less than or equal to 0.")

    # move computation axis to end
    x = moveaxis(a, axis, -1)

    # check that there are enough samples
    if w_len > x.shape[-1]:
        raise ValueError("Window length is larger than the computation axis.")

    rmean = _extensions.moving_mean(x, w_len, skip, trim)

    # move computations axis back to original place and return
    return moveaxis(rmean, -1, axis)

                    



def apply_resample(*, time, goal_fs=None, time_rs=None, data=(), indices=(), aa_filter=True, fs=None):
    
    def resample(x, factor, t, t_rs):
        if (int(factor) == factor) and (factor>1):
            # in case that t_rs is provided and ends earlier than t
            n = nonzero(t<=t_rs[-1])[0][-1] + 1
            return 0   (x[: n : int(factor)], )
        else:
            if x.ndim==1:
                return (interp(t_rs, t, x), )
            elif x.ndim==2:
                xrs = zeros((t_rs.size, x.shape[1]), dtype=float64)
                for j in range(x.shape[1]):
                    xrs[:, j] = interP(t_rs, t, x[:,j])
                return (xrs, )
            
    
    if fs is None:
        fs = 1 / mean(diff(time[:5000]))

    if time_rs is None and goal_fs is None:
        raise ValueError("One of `time_rs` or `goal_fs` is required.")
    
    # get resampled time if necessary
    if time_rs is None:
        if int(fs / goal_fs) == fs / goal_fs and goal_fs < fs:
            time_rs = time[:: int(fs / goal_fs)]
        else:
            # round-about way, but need to prevent start>>>>>>>>>step
            time_rs = arange(0, (time[-1] - time[0]) + 0.5 / goal_fs, 1 / goal_fs) + time[0]
    else:
        goal_fs = 1 / mean(diff(time_rs[:5000]))
        # prevent t_rs from extrapolating
        time_rs = time_rs[time_rs <= time[-1]]

    # AA filter, if necessary
    if (fs / goal_fs) >= 1.0:
        # adjust the cutoff frequency based on if we are decimating or interpolating
        # decimation cutoff comes from scipy.decimate
        wn = 0.8 / (fs / goal_fs) if int(fs / goal_fs) == fs / goal_fs else goal_fs / fs
        sos = cheby1(8, 0.05, wn, output="sos")
    else:
        aa_filter = False

    # resample data
    data_rs = ()

    for dat in data:
        if dat is None:
            data_rs += (None,)
        elif dat.ndim in [1, 2]:
            data_to_rs = sosfiltfilt(sos, dat, axis=0) if aa_filter else dat
            data_rs += resample(data_to_rs, fs / goal_fs, time, time_rs)
        else:
            raise ValueError("Data dimension exceeds 2, or data not understood.")

    # resampling indices
    indices_rs = ()
    for idx in indices:
        if idx is None:
            indices_rs += (None,)
        elif idx.ndim == 1:
            indices_rs += (
                around(interp(time[idx], time_rs, arange(time_rs.size))).astype(int_),
            )
        elif idx.ndim == 2:
            indices_rs += (zeros(idx.shape, dtype=int_),)
            for i in range(idx.shape[1]):
                indices_rs[-1][:, i] = around(
                    interp(
                        time[idx[:, i]], time_rs, arange(time_rs.size)
                    )  # cast to in on insert
                )

    ret = (time_rs,)
    if data_rs != ():
        ret += (data_rs,)
    if indices_rs != ():
        ret += (indices_rs,)

    return ret










def get_activity_counts(fs, time, accel, epoch_seconds=60):
    # Down-sample to 30 Hz
    time_ds, (acc_ds, ) = apply_resample(
        goal_fs=30.0,
        time=time,
        data=(accel,)
        aa_filter=True,
        fs=fs
    )

    # Filter data
    zi = lfilter_zi(input_coef, output_coef).reshape((-1, 1))

    acc_bpf, _ = lfilter(
        input_coef,
        output_coef,
        acc_ds,
        zi=repeat(zi, acc_ds.shape[1], axis=1) * acc_ds[0],
        axis=0,
    )

    # Scale the data
    acc_bpf *= (3 / 4096) / (2.6 / 256) * 237.5

    # rectify
    acc_trim = abs(acc_bpf)

    # trim
    acc_trim[acc_trim<4] = 0
    acc_trim = floor(minimum(acc_trim, 128))

    # downsample
    acc_10hz = moving_mean(acc_trim, 3, 3, trim=True, axis=0)

    # get counts
    block_size = epoch_seconds * 10 # 1 min

    # moving sum
    epoch_counts = moving_mean(acc_10hz, block_size, block_size, trim=True, axis=0)
    epoch_counts *= block_size # remove mean part to get sum back

    return epoch_counts

