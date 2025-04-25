import numpy as np
from warnings import warn
from numpy import argmax, abs, mean, cos, arcsin, sign, zeros_like
import matplotlib.pyplot as plt


# Simulate 10 seconds of statiobary data at 100 Hz
N = 1000 # 10 sec x 100 samples/sec
accel = np.zeros((N, 3))

# Simulate gravity mostly along Z (index 2), small noise on other axes
accel[ : , 0] = 0.01*np.random.randn(N) # X axis (AP)
accel[ : , 1] = 0.01*np.random.randn(N) # Y axis (ML)
accel[ : , 2] = 1+ 0.01*np.random.randn(N) # Z axis (VERT)





def correct_accelerometer_orientation(accel, v_axis=None, ap_axis=None):

    if v_axis is None:
        v_axis = argmax(abs(mean(accel, axis=0)))
    else:
        if not (0<=v_axis<3):
            raise ValueError("v_axis must be in {0, 1, 2}")
    
    if ap_axis is None:
        ap_axis, ml_axis = [i for i in range(3) if i != v_axis]
    else:
        if not (0<=ap_axis<3):
            raise ValueError("ap_axis must be in {0, 1, 2}")
        ml_axis = [i for i in range(3) if i not in [v_axis, ap_axis]][0]
    
    s_theta_a = mean(accel[:, ap_axis])
    s_theta_m = mean(accel[:, ml_axis])

    # make sure the theta values (average accel) are in range < 1G
    if s_theta_a < -1 or s_theta_a > 1 or s_theta_m < -1 or s_theta_m > 1:
        warn("Accel correction angles outside possible range [-1,1]. Not correcting.")
        return accel # original
    

    c_theta_a = cos(arcsin(s_theta_a))
    c_theta_m = cos(arcsin(s_theta_m))

    v_sign = sign(mean(accel[:, v_axis]))

    co_accel = zeros_like(accel)

    # correct ap axis
    co_accel[:, ap_axis] = ( 
        accel[:, ap_axis] * c_theta_a - v_sign * accel[:, v_axis] * s_theta_a
        )
    # correct vertical axis provisional
    co_accel[:, v_axis] = (
        v_sign * accel[: ,ap_axis] * s_theta_a + accel[:, v_axis] * c_theta_a
        )
    # correct ml axis acceleration
    co_accel[:, ml_axis] = (
        accel[:, ml_axis] * c_theta_m - v_sign * co_accel[:, v_axis] * s_theta_m
    )
    # final correction for vertical axis
    co_accel[:, v_axis] = (
        v_sign * accel[:, ml_axis] * s_theta_m + co_accel[:, v_axis] * c_theta_m
    )
    return co_accel








# Call the correction function
corrected = correct_accelerometer_orientation(accel)

print("Means BEFORE correction: ", np.mean(accel, axis=0))
print("Means AFTER correction: ", np.mean(corrected, axis=0))





# Simulate tilted IMU data: assume device is tilted 20 degrees forward (about the ML axis)
tilt_angle_deg = 20
tilt_angle_rad = np.radians(tilt_angle_deg)

# Rotation matrix for tilt about Y-axis (ML axis), rotating in the X-Z plane
R_tilt = np.array([
    [np.cos(tilt_angle_rad), 0, np.sin(tilt_angle_rad)],
    [0, 1, 0],
    [-np.sin(tilt_angle_rad), 0, np.cos(tilt_angle_rad)]
])

# Apply tilt to gravity vector (0, 0, 1) across all samples
gravity = np.tile([0, 0, 1.0], (N, 1))  # original vertical acceleration
tilted_accel = gravity @ R_tilt.T  # rotate the gravity vector

# Add some random noise to simulate real measurements
tilted_accel += 0.01 * np.random.randn(*tilted_accel.shape)


# Apply correction to tilted data
corrected_tilted = correct_accelerometer_orientation(tilted_accel)


# Plot original vs corrected
fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

labels = ['X (AP)', 'Y (ML)', 'Z (Vertical)']


for i in range(3):
    axs[i].plot(tilted_accel[:, i], label='Tilted', alpha=0.6)
    axs[i].plot(corrected_tilted[:, i], label='Corrected', linestyle='--')
    axs[i].set_ylabel(labels[i])
    axs[i].legend(loc='upper right')
    axs[i].grid(True)

axs[2].set_xlabel('Sample Index (Time)')
plt.suptitle('Tilted Accelerometer Data Before and After Correction (20° Tilt)')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()





# Simulate simple walking-like pattern: vertical bouncing + AP swinging
t = np.linspace(0, 10, N)  # 3 seconds

# Simulate AP (X) forward-back oscillation, ML (Y) side sway, vertical (Z) bounce
walking_accel = np.zeros((N, 3))
walking_accel[:, 0] = 0.2 * np.sin(2 * np.pi * 1.5 * t)  # AP swing at ~1.5 Hz
walking_accel[:, 1] = 0.05 * np.sin(2 * np.pi * 3.0 * t + np.pi / 4)  # ML sway
walking_accel[:, 2] = 1.0 + 0.1 * np.abs(np.sin(2 * np.pi * 1.5 * t))  # vertical bounce

# Add some noise
walking_accel += 0.01 * np.random.randn(*walking_accel.shape)

# Apply correction
corrected_walk = correct_accelerometer_orientation(walking_accel)

# Plot results
fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

for i in range(3):
    axs[i].plot(walking_accel[:, i], label='Original (Walking)', alpha=0.6)
    axs[i].plot(corrected_walk[:, i], label='Corrected', linestyle='--')
    axs[i].set_ylabel(labels[i])
    axs[i].legend(loc='upper right')
    axs[i].grid(True)

axs[2].set_xlabel('Sample Index (Time)')
plt.suptitle('Simulated Walking Accelerometer Data Before and After Correction')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
