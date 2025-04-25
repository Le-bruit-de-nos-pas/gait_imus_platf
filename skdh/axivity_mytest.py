from numpy import ascontiguousarray

from skdh.base import handle_process_returns
from skdh.io.base import check_input_file, handle_naive_timestamps, BaseIO
from skdh.io._extensions import read_axivity

class UnexpectedAxesError(Exception):
    pass

class ReadCwa(BaseIO):
    # reader = ReadCwa()
    # reader.predict('example.cwa)

    def __init__(self, * , trim_keys=None, ext_error="warn"):
        super().__init__(
            trim_keys=trim_keys,
            ext_error=ext_error
        )

        self.trim_keys = trim_keys
        
        if ext_error.lower() in ["warn", "raise", "skip"]:
            self.ext_error = ext_error.lower()
        else:
            raise ValueError("`ext_error` must be one of 'raise', 'warn', 'skip'.")
        
    @handle_process_returns(results_to_kwargs=True)
    @check_input_file(".cwa")

    def predict(self, *, file, tz_name=None, **kwargs):
        super().predict(
            expect_days=False, expect_wear=False, file=file, tz_name=tz_name, **kwargs 
        )

        fs, n_bad_samples, imudata, ts, temperature = read_axivity(str(file))

        # end = None if n_bad_samples == 0, else -n_bad_samples
        end=None
        num_axes = imudata.shape[1]
        gyr_axes = mag_axes = None
        
        if num_axes==3:
            acc_axes = slice(None)
        elif num_axes==6:
            gyr_axes = slice(3)
            acc_axes=slice(3,6)
        elif num_axes==9:
            gyr_axes = slice(3)
            acc_axes=slice(3,6)
            mag_axes=slice(6,9)
        else:
            raise UnexpectedAxesError("Unexpected number of axes in the IMU data")

        results= {
            self._time: handle_naive_timestamps(
                ts[:end], is_local=True, tz_name=tz_name
            ),
            self._temp: temperature[:end],
        }

        if acc_axes is not None:
            results[self._acc] = ascontiguousarray(imudata[:end, acc_axes])
        if gyr_axes is not None:
            results[self._gyro] = ascontiguousarray(imudata[:end, gyr_axes])
        if mag_axes is not None:  # pragma: no cover :: don't have data to test this
            results[self._mag] = ascontiguousarray(imudata[:end, mag_axes])

        if self.trim_keys is not None:
            results = self.trim_data(
                *self.trim_keys,
                tz_name,
                kwargs,
                **results,  # contains the time array/argument
            )

        results["fs"] = fs

        return results
