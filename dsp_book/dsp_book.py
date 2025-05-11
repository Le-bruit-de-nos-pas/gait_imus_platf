def gregory_pi_approximation(n_terms):
    pi_over_4 = 0
    for n in range(n_terms+1):
        term = ((-1)**n) / (2*n+1)
        pi_over_4 += term
    pi_approx = 4*pi_over_4
    return pi_approx       


approx_pi = gregory_pi_approximation(1000)

print(f"Approximation = {approx_pi} ") 









import numpy as np
import matplotlib.pyplot as plt

# Time array: simulate continuous time with high resolution
t = np.linspace(0, 1, 1000) # 1 sec, 1000 samples

# Generate a square wave that switches every 0.1s
frequency = 5 # 5 Hz, 0.2s period
signal = 2.5 * (1+ np.sign(np.sin(2*np.pi*frequency*t)))

plt.figure(figsize=(10,4))
plt.plot(t, signal, label='Square wave signal (0v/5v)', color='blue')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (V)')

plt.grid(True)

plt.ylim(-1, 6)
plt.legend()
plt.tight_layout()
plt.show()








import numpy as np
import matplotlib.pyplot as plt

# Time array: simulate continuous time with high resolution
t = np.linspace(0, 1, 1000)  # 1 second, 1000 samples

# Generate a square wave that switches every 0.1s
frequency = 5  # 5 Hz → 0.2s period
signal = 2.5 * (1 + np.sign(np.sin(2 * np.pi * frequency * t)))  # Outputs 0 or 5

# Plot
plt.figure(figsize=(10, 4))
plt.plot(t, signal, label='Square Wave Signal (0V / 5V)', color='blue')
plt.title('Continuous-Time Signal with Finite Amplitude Values')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (V)')
plt.grid(True)
plt.ylim(-1, 6)
plt.legend()
plt.tight_layout()
plt.show()





import math

# Compute π using the inverse sine of 1.0
pi = 2 * math.asin(1.0)

print(f"π calculated using 2 * asin(1.0): {pi}")





import matplotlib.pyplot as plt
import numpy as np

# Create a range of x values (excluding 0 to avoid undefined 0^0)
x = np.linspace(0.1, 10, 100)
y = x ** 0

# Plot x^0 = 1
plt.plot(x, y, label='x^0', color='blue')
plt.axhline(1, color='gray', linestyle='--', linewidth=1)
plt.title('Graph of x^0 for x ≠ 0')
plt.xlabel('x')
plt.ylabel('x^0')
plt.grid(True)
plt.legend()
plt.show()






import numpy as np
import matplotlib.pyplot as plt

# Settings
fs = 100  # Sample frequency (Hz)
n = np.arange(0, 20)  # 20 sample points
Ts = 1 / fs  # Sample period

# Frequencies
f0_a = fs / 2   # Case (a)
f0_b = fs / 4   # Case (b)
f0_c = 0        # Case (c)

# Sequences
x_a = np.cos(2 * np.pi * f0_a * n * Ts)  # Should simplify to cos(pi * n)
x_b = np.cos(2 * np.pi * f0_b * n * Ts)  # cos(pi/2 * n)
x_c = np.cos(2 * np.pi * f0_c * n * Ts)  # cos(0)

# Plotting
plt.figure(figsize=(12, 6))

# (a) fo = fs/2
plt.subplot(3, 1, 1)
plt.stem(n, x_a)
plt.title("(a) fo = fs / 2 → x[n] = cos(πn)")
plt.ylabel("Amplitude")
plt.grid(True)

# (b) fo = fs/4
plt.subplot(3, 1, 2)
plt.stem(n, x_b)
plt.title("(b) fo = fs / 4 → x[n] = cos(πn/2)")
plt.ylabel("Amplitude")
plt.grid(True)

# (c) fo = 0
plt.subplot(3, 1, 3)
plt.stem(n, x_c)
plt.title("(c) fo = 0 → x[n] = 1")
plt.xlabel("Sample index n")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()
plt.show()






# Define the three sine sequences
s1 = np.sin(np.pi * n)         # fs/2
s2 = np.sin(np.pi * n / 2)     # fs/4
s3 = np.zeros_like(n)          # 0 Hz

# Create subplots
fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

# Plot (a) fo = fs/2
axs[0].stem(n, s1)
axs[0].set_title('(a) $x[n] = \sin(\pi n)$ (fo = fs/2)')
axs[0].set_ylabel('Amplitude')
axs[0].grid(True)

# Plot (b) fo = fs/4a+
axs[1].stem(n, s2)
axs[1].set_title('(b) $x[n] = \sin(\pi n / 2)$ (fo = fs/4)')
axs[1].set_ylabel('Amplitude')
axs[1].grid(True)

# Plot (c) fo = 0
axs[2].stem(n, s3)
axs[2].set_title('(c) $x[n] = 0$ (fo = 0 Hz)')
axs[2].set_ylabel('Amplitude')
axs[2].set_xlabel('n')
axs[2].grid(True)

plt.tight_layout()
plt.show()







from scipy.signal import lfilter
import numpy as np

# Define a simple FIR low-pass filter (e.g., moving average filter)
# 5-point moving average
b = np.ones(5) / 5  # Filter coefficients
a = [1]             # FIR filter (no feedback)

# Input signals: cosine at 0 Hz, fs/2, fs/4
cos_0Hz = np.cos(2 * np.pi * 0 * n / len(n))        # DC
cos_fs2 = np.cos(np.pi * n)                         # fs/2
cos_fs4 = np.cos(np.pi * n / 2)                     # fs/4

# Filter the signals
y_0Hz = lfilter(b, a, cos_0Hz)
y_fs2 = lfilter(b, a, cos_fs2)
y_fs4 = lfilter(b, a, cos_fs4)

# Plot
fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

axs[0].stem(n, y_0Hz)
axs[0].set_title("Filtered cos(0 Hz) — Low-pass output (should pass)")
axs[0].grid(True)

axs[1].stem(n, y_fs2)
axs[1].set_title("Filtered cos(fs/2) — Low-pass output (should attenuate)")
axs[1].grid(True)

axs[2].stem(n, y_fs4)
axs[2].set_title("Filtered cos(fs/4) — Low-pass output (partial pass)")
axs[2].grid(True)
axs[2].set_xlabel("n")

plt.tight_layout()
plt.show()









# High-pass filter: flip the sign of every other tap to convert low-pass to high-pass
# This is done by multiplying by (-1)^n = cos(pi*n)
hp_filter = b * np.cos(np.pi * np.arange(len(b)))

# Filter the signals using the high-pass filter
y_hp_0Hz = lfilter(hp_filter, a, cos_0Hz)
y_hp_fs2 = lfilter(hp_filter, a, cos_fs2)
y_hp_fs4 = lfilter(hp_filter, a, cos_fs4)

# Plot
fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

axs[0].stem(n, y_hp_0Hz)
axs[0].set_title("High-pass Filtered cos(0 Hz) — should be attenuated")
axs[0].grid(True)

axs[1].stem(n, y_hp_fs2)
axs[1].set_title("High-pass Filtered cos(fs/2) — should be passed")
axs[1].grid(True)

axs[2].stem(n, y_hp_fs4)
axs[2].set_title("High-pass Filtered cos(fs/4) — partial pass")
axs[2].grid(True)
axs[2].set_xlabel("n")

plt.tight_layout()
plt.show()






# Define sample rate and time indices
fs = 2500  # Hz
n = np.arange(0, 40)  # sample indices
m = np.sin(0.8 * np.pi * n)  # modulation signal

# Plot the modulating signal
plt.figure(figsize=(10, 3))
plt.stem(n, m)
plt.title("Modulating Signal m(n) = sin(0.8πn), Frequency = 1000 Hz")
plt.xlabel("Sample Index n")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()






import numpy as np

# Define a sample input sequence x[n] (e.g., length 16)
x = np.array([2, 4, 1, 3, 5, 2, 6, 4, 3, 7, 0, 1, 2, 3, 1, 0])

# Set N = 6 for the running sum
N = 6

# Compute y(9) as the sum of x[4] through x[9]
y_9 = np.sum(x[4:10])  # x[4] to x[9] inclusive

# Show the values involved and the result
x_values = x[4:10]  # x[4], x[5], ..., x[9]

x_values, y_9







# Compute the full running sum output sequence y[n]
# We'll start from index N-1 so we have enough samples for each sum
y = np.array([np.sum(x[n-N+1:n+1]) for n in range(N-1, len(x))])


# Create corresponding time indices for y[n]
n_y = np.arange(N-1, len(x))

# Plot the result
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.stem(n_y, y)
plt.title(f'{N}-Point Running Sum Output y[n]')
plt.xlabel('n')
plt.ylabel('y[n]')
plt.grid(True)
plt.tight_layout()
plt.show()






import numpy as np
import matplotlib.pyplot as plt

# Define parameters
fs = 100  # Sampling frequency in Hz
ts = 1 / fs  # Sampling period
fo = 5       # Frequency of the cosine waves
phi = np.pi / 2  # Phase shift in radians
n = np.arange(0, 50)  # 50 sample points
t = n * ts  # time vector

# Original expression
x_original = np.cos(2 * np.pi * fo * t + phi) + np.cos(2 * np.pi * fo * t)

# Derived expression
x_derived = 2 * np.cos(2 * np.pi * fo * t + phi / 2) * np.cos(phi / 2)

# Plotting
plt.figure(figsize=(10, 4))
plt.stem(n, x_original, linefmt='b-', markerfmt='bo', basefmt=' ', label='Original Expression')
plt.plot(n, x_derived, 'r--', linewidth=1.5, label='Derived Expression')
plt.title('Comparison of Original and Derived Expressions for x(n)')
plt.xlabel('Sample Index n')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()





# Define alpha values from -π/2 to π/2
alpha = np.linspace(-np.pi/2, np.pi/2, 500)

# Define x = α and y = sin(α)
x = alpha
y = np.sin(alpha)

# Plotting
plt.figure(figsize=(8, 4))
plt.plot(alpha, x, label='x = α', linestyle='--', color='gray')
plt.plot(alpha, y, label='y = sin(α)', color='blue')
plt.title('Comparison of x = α and y = sin(α)')
plt.xlabel('α (radians)')
plt.ylabel('Value')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()





import numpy as np
import matplotlib.pyplot as plt

# Define original signal x(n) = sin(pi/4 * n)
n = np.arange(0, 16)
x = np.sin((np.pi / 4) * n)

# Decimated version of x(n): y(m) = x(2m)
m = np.arange(0, len(n) // 2)
y = x[2 * m]

# Shifted signal x_shift(n) = x(n - 1)
x_shift = np.sin((np.pi / 4) * (n - 1))

# Decimated version of x_shift(n): y_shift(m) = x_shift(2m) = x(2m - 1)
y_shift = np.sin((np.pi / 4) * (2 * m - 1))

# Plotting
plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.stem(n, x, basefmt=" ")
plt.title("Original Signal x(n)")
plt.xlabel("n")
plt.ylabel("x(n)")

plt.subplot(3, 1, 2)
plt.stem(m, y, linefmt='g-', markerfmt='go', basefmt=" ")
plt.title("Decimated Signal y(m) = x(2m)")
plt.xlabel("m")
plt.ylabel("y(m)")

plt.subplot(3, 1, 3)
plt.stem(m, y_shift, linefmt='r-', markerfmt='ro', basefmt=" ")
plt.title("Decimated Shifted Signal y_shift(m) = x(2m - 1)")
plt.xlabel("m")
plt.ylabel("y_shift(m)")

plt.tight_layout()
plt.show()





import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

# Create an example input signal: x(n) = sin(2πfn), sampled at discrete intervals
n = np.arange(0, 50)
x = np.sin(2 * np.pi * 0.05 * n)

# Define two simple FIR filters (impulse responses)
h1 = np.array([0.5, 0.5])  # Moving average filter
h2 = np.array([1, -1])     # Differentiator filter

# Apply H1 then H2
x_h1 = np.convolve(x, h1, mode='full')
y1 = np.convolve(x_h1, h2, mode='full')

# Apply H2 then H1
x_h2 = np.convolve(x, h2, mode='full')
y2 = np.convolve(x_h2, h1, mode='full')

# Truncate or pad to same length for comparison (they will be same length here)
min_len = min(len(y1), len(y2))
y1 = y1[:min_len]
y2 = y2[:min_len]
n_out = np.arange(min_len)

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(n_out[:len(x)], x, 'b-', label='(x)')
plt.plot(n_out, y1, 'b-', label='H2(H1(x))')
plt.plot(n_out, y2, 'r--', label='H1(H2(x))')
plt.title('Commutative Property of LTI Systems')
plt.xlabel('n')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()






import numpy as np
import matplotlib.pyplot as plt

# Define n range
n = np.arange(0, 20)

# Unit impulse input
x = np.zeros_like(n)
x[0] = 1  # unit impulse at n = 0

# Initialize outputs
y_comb = np.zeros_like(n)
y_int = np.zeros_like(n)
y_leaky = np.zeros_like(n)
y_diff = np.zeros_like(n)

# Leaky integrator parameter
A = 0.5

# Generate outputs
for i in range(len(n)):
    # Comb filter: yC(n) = x(n) - x(n-4)
    if i >= 4:
        y_comb[i] = x[i] - x[i - 4]
    else:
        y_comb[i] = x[i]

    # Integrator: yI(n) = x(n) + yI(n-1)
    if i == 0:
        y_int[i] = x[i]
    else:
        y_int[i] = x[i] + y_int[i - 1]

    # Leaky integrator: yLI(n) = A*x(n) + (1-A)*yLI(n-1)
    if i == 0:
        y_leaky[i] = A * x[i]
    else:
        y_leaky[i] = A * x[i] + (1 - A) * y_leaky[i - 1]

    # Differentiator: yD(n) = 0.5*x(n) - 0.5*x(n-2)
    if i >= 2:
        y_diff[i] = 0.5 * x[i] - 0.5 * x[i - 2]
    elif i == 0:
        y_diff[i] = 0.5 * x[i]
    else:
        y_diff[i] = 0.0
        
# Plot all impulse responses
plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
plt.stem(n, y_comb, basefmt=" ")
plt.title("Impulse Response of 4th-Order Comb Filter")
plt.xlabel("n")
plt.ylabel("y_C(n)")

plt.subplot(4, 1, 2)
plt.stem(n, y_int, basefmt=" ")
plt.title("Impulse Response of Integrator")
plt.xlabel("n")
plt.ylabel("y_I(n)")

plt.subplot(4, 1, 3)
plt.stem(n, y_leaky, basefmt=" ")
plt.title("Impulse Response of Leaky Integrator (A = 0.5)")
plt.xlabel("n")
plt.ylabel("y_LI(n)")

plt.subplot(4, 1, 4)
plt.stem(n, y_diff, basefmt=" ")
plt.title("Impulse Response of Differentiator")
plt.xlabel("n")
plt.ylabel("y_D(n)")

plt.tight_layout()
plt.show()





import numpy as np
import matplotlib.pyplot as plt

# Time range
n = np.arange(0, 20)

# Unit impulse input
x = np.zeros_like(n)
x[0] = 1  # Impulse at n = 0

# Initialize impulse responses
y_comb = np.zeros_like(n, dtype=float)
y_int = np.zeros_like(n, dtype=float)
y_leaky = np.zeros_like(n, dtype=float)
y_diff = np.zeros_like(n, dtype=float)

# Leaky integrator coefficient
A = 0.5

# Calculate impulse responses
for i in range(len(n)):
    # Comb Filter
    if i >= 4:
        y_comb[i] = x[i] - x[i - 4]
    else:
        y_comb[i] = x[i]

    # Integrator
    if i == 0:
        y_int[i] = x[i]
    else:
        y_int[i] = x[i] + y_int[i - 1]

    # Leaky Integrator
    if i == 0:
        y_leaky[i] = A * x[i]
    else:
        y_leaky[i] = A * x[i] + (1 - A) * y_leaky[i - 1]

    # Differentiator
    if i >= 2:
        y_diff[i] = 0.5 * x[i] - 0.5 * x[i - 2]
    elif i == 0:
        y_diff[i] = 0.5 * x[i]
    else:
        y_diff[i] = 0.0

# Step responses: cumulative sum of impulse responses
s_comb = np.cumsum(y_comb)
s_int = np.cumsum(y_int)
s_leaky = np.cumsum(y_leaky)
s_diff = np.cumsum(y_diff)

# Plot all step responses
plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
plt.stem(n, s_comb, basefmt=" ")
plt.title("Step Response of 4th-Order Comb Filter")
plt.xlabel("n")
plt.ylabel("ystep_C(n)")

plt.subplot(4, 1, 2)
plt.stem(n, s_int, basefmt=" ")
plt.title("Step Response of Integrator")
plt.xlabel("n")
plt.ylabel("ystep_I(n)")

plt.subplot(4, 1, 3)
plt.stem(n, s_leaky, basefmt=" ")
plt.title("Step Response of Leaky Integrator (A = 0.5)")
plt.xlabel("n")
plt.ylabel("ystep_LI(n)")

plt.subplot(4, 1, 4)
plt.stem(n, s_diff, basefmt=" ")
plt.title("Step Response of Differentiator")
plt.xlabel("n")
plt.ylabel("ystep_D(n)")

plt.tight_layout()
plt.show()






import numpy as np
import matplotlib.pyplot as plt

# Define the time vector
t = np.linspace(0, 2, 1000)  # 2 seconds, fine resolution
f0 = 1  # Fundamental frequency in Hz
A = 1  # Amplitude

# Original squarewave using first several odd harmonics
def squarewave(t, N):
    """Approximate squarewave using N odd harmonics."""
    s = np.zeros_like(t)
    for n in range(1, N+1, 2):  # Only odd harmonics
        s += (1/n) * np.sin(2 * np.pi * n * f0 * t)
    return (4 * A / np.pi) * s

# Case 1: Full squarewave with first 7 harmonics (1 Hz to 13 Hz)
full_square = squarewave(t, 13)

# Case 2: Remove the first harmonic (i.e., start from 3rd)
filtered_square = squarewave(t, 13) - (4 * A / np.pi) * (1/1) * np.sin(2 * np.pi * f0 * t)

# Plotting
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(t, full_square, label='Original Squarewave (up to 13 Hz)', color='blue')
plt.title('Original Squarewave (Fundamental and Odd Harmonics up to 13 Hz)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t, filtered_square, label='Filtered Squarewave (No 1 Hz)', color='red')
plt.title('Squarewave with Fundamental Removed (Starting from 3rd Harmonic)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()









import numpy as np
import matplotlib.pyplot as plt

# Set up clock positions
def minute_hand_angle(minutes):
    """Return angle in radians for given minutes on a clock (0 at top, clockwise positive)."""
    return np.deg2rad(90 - 6*minutes) # 6 degrees per minute, 90 degree is up

print(minute_hand_angle(0))
print(minute_hand_angle(30))

# Time steps in minutes
times_aliased = np.arange(0, 5*55, 55) # every 55 min (5 steps)
times_nyquist = np.arange(0, 5*20, 20) # every 20 min (3 rotations/hour) (5 steps)

print(times_aliased)
print(times_nyquist)

# Convert to angles
# Convert to angles
angles_aliased = minute_hand_angle(times_aliased % 60)
angles_nyquist = minute_hand_angle(times_nyquist % 60)

print(angles_aliased)
print(angles_nyquist)


# Plot clock face
def plot_clock(ax, angles, title, color):
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.axis('off')

    # Draw clock circle
    clock_circle = plt.Circle( (0,0), 1, fill=False, linewidth=2)
    ax.add_artist(clock_circle)

    # Draw minutes amrkers
    for i in range(12):
        angle = np.deg2rad(90 - i*30)
        x = 0.9 * np.cos(angle)
        y = 0.9 * np.sin(angle)
        ax.text(x, y, str(i if i !=0 else 12), ha='center', va='center', fontsize=12)

    # Plot minute hand positions
    for angle in angles:
        x = 0.8 * np.cos(angle)
        y = 0.8 * np.sin(angle)
        ax.plot([0,x], [0,y], color=color, linewidth=2)
        ax.plot(x,y,'o', color=color)


# Create plots
fig, axs = plt.subplots(1,2,figsize=(12,6))


plot_clock(axs[0], angles_aliased, 'Photos Every 55 min (Aliased, appears CCW)', 'red')
plot_clock(axs[1], angles_nyquist, 'Photos Every 20 min (Proper CW)', 'green')

plt.tight_layout()
plt.show()






import matplotlib.animation as animation

fig, ax = plt.subplots(figsize=(6,6))

# Plot static elements (clock face)
def setup_clock_face(ax):
    ax.set_aspect('equal')
    ax.set_xlim(-1.2,1.2)
    ax.set_ylim(-1.2,1.2)
    ax.axis('off')
    clock_circle = plt.Circle((0,0), 1, fill=False, linewidth=2)
    ax.add_artist(clock_circle)

    for i in range(12):
        angle = np.deg2rad(90 - i*30)
        x = 0.9 * np.cos(angle)
        y = 0.9 * np.sin(angle)
        ax.text(x, y, str(i if i != 0 else 12), ha='center', va='center', fontsize=10)


setup_clock_face(ax)
hand_line, = ax.plot([], [], 'o-', lw=3, color='blue')

# Generate frame angles for 20-minute interval sampling (proper CW motion)
angles = angles_nyquist

# Initialization function for animation
def init():
    hand_line.set_data([], [])
    return hand_line, 

# Animation update function
def update(frame):
    angle = angles[frame % len(angles)]
    x = [0, 0.8 * np.cos(angle)]
    y = [0, 0.8 * np.sin(angle)]
    hand_line.set_data(x, y)
    return hand_line,

ani = animation.FuncAnimation(fig, update, frames=len(angles), init_func=init,
                              blit=True, interval=1000, repeat=True)

plt.close(fig)
ani







from IPython.display import HTML
HTML(ani.to_jshtml())





# Generate frame angles for 55-minute interval sampling (aliased, appears CCW)
angles_alias = angles_aliased

# Re-create figure and axis for aliased animation
fig_alias, ax_alias = plt.subplots(figsize=(6, 6))

setup_clock_face(ax_alias)
hand_line_alias, = ax_alias.plot([], [], 'o-', lw=3, color='red')

# Initialization function
def init_alias():
    hand_line_alias.set_data([], [])
    return hand_line_alias,

# Animation update function
def update_alias(frame):
    angle = angles_alias[frame % len(angles_alias)]
    x = [0, 0.8 * np.cos(angle)]
    y = [0, 0.8 * np.sin(angle)]
    hand_line_alias.set_data(x, y)
    return hand_line_alias,

# Create aliased animation
ani_alias = animation.FuncAnimation(fig_alias, update_alias, frames=len(angles_alias),
                                    init_func=init_alias, blit=True, interval=1000, repeat=True)

plt.close(fig_alias)  # Prevent static plot
ani_alias






from IPython.display import HTML
HTML(ani_alias.to_jshtml())




# Visualize how sampling rate affects interpretation of frequency

# Continuous-time signal: a 5 Hz sine wave sampled at different rates
t_fine = np.linspace(0, 1, 1000)  # high-resolution time axis
x_cont = np.sin(2 * np.pi * 5 * t_fine)  # 5 Hz sine wave

# Sample at two different rates
fs1 = 20  # Hz, just enough to satisfy Nyquist
fs2 = 50  # Hz, oversampled

t_samples_fs1 = np.arange(0, 1, 1/fs1)
x_samples_fs1 = np.sin(2 * np.pi * 5 * t_samples_fs1)

t_samples_fs2 = np.arange(0, 1, 1/fs2)
x_samples_fs2 = np.sin(2 * np.pi * 5 * t_samples_fs2)

# Plotting
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# First subplot: fs = 20 Hz
axs[0].plot(t_fine, x_cont, label="Original Signal (5 Hz)", color='gray')
axs[0].stem(t_samples_fs1, x_samples_fs1, linefmt='C0-', markerfmt='C0o', basefmt=" ")
axs[0].set_title("Sampling at 20 Hz (just above Nyquist)")
axs[0].set_ylabel("Amplitude")
axs[0].grid(True)
axs[0].legend()

# Second subplot: fs = 50 Hz
axs[1].plot(t_fine, x_cont, label="Original Signal (5 Hz)", color='gray')
axs[1].stem(t_samples_fs2, x_samples_fs2, linefmt='C1-', markerfmt='C1o', basefmt=" ")
axs[1].set_title("Sampling at 50 Hz (oversampled)")
axs[1].set_xlabel("Time (seconds)")
axs[1].set_ylabel("Amplitude")
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.show()






import numpy as np
import matplotlib.pyplot as plt

# Define signal parameters
f0 = 1000 # Hz frequency of the continuous sinewave
f2 = 2000# Hz sampling rate (Nyquist rate = fs = 2*f0)
phi = np.pi/2 # phase shift (90 degress to force zero crossings at sampling)

# Define time vector
t_cont = np.linspace(0, 0.03, 5000) # 4 cycles
x_cont = np.cos(2*np.pi * f0 * t_cont + phi)

# define sample points
n = np.arange(0, 4) # 4 samples
t_samp = n/fs # 4 / 2000
x_samp = np.cos(2*np.pi * f0 * t_samp + phi)

print("sampled values:", x_samp)

# Plot
plt.figure(figsize=(10, 4))
plt.plot(t_cont, x_cont, label="Continuous Signql")
plt.stem(t_samp, x_samp, linefmt="r-", markerfmt="ro", basefmt="k-", label="Sampled points")
plt.title("Sampling a sinusoid at nyquist rate with phase shift")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Given values
fs = 160  # Sampling rate in Hz
n = np.arange(0, 16)  # Sample indices
t_cont = np.linspace(0, 15 / fs, 1000)  # Continuous time for smooth curves

# Frequencies to test
frequencies = [20, 140, 180]  # Hz

# Plotting
plt.figure(figsize=(12, 6))

for i, f0 in enumerate(frequencies, 1):
    # Continuous signal
    x_cont = np.sin(2 * np.pi * f0 * t_cont)
    # Discrete signal
    x_disc = np.sin(2 * np.pi * f0 * n / fs)
    
    plt.subplot(1, 3, i)
    plt.plot(t_cont, x_cont, label=f'Continuous ({f0} Hz)', color='gray', alpha=0.7)
    plt.stem(n / fs, x_disc, linefmt='C0-', markerfmt='C0o', basefmt='k-', label='Sampled')
    plt.title(f'$f_0 = {f0}$ Hz')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.suptitle("Sampling sinusoids at $f_s = 160$ Hz", fontsize=14, y=1.05)
plt.show()



from scipy.fft import fft, fftfreq

# Parameters for FFT
N = 64  # number of samples for FFT
t_fft = np.arange(N) / fs  # time vector for sampled signal

# Prepare plot
plt.figure(figsize=(12, 8))

for i, f0 in enumerate(frequencies, 1):
    # Discrete signal sampled at fs
    x_n = np.sin(2 * np.pi * f0 * t_fft)
    
    # Compute FFT
    X_f = fft(x_n)
    freqs = fftfreq(N, 1/fs)
    
    # Plot magnitude spectrum (only positive freqs)
    plt.subplot(3, 1, i)
    plt.stem(freqs[:N//2], np.abs(X_f[:N//2]), basefmt="k-")
    plt.title(f'FFT Magnitude Spectrum for $f_0 = {f0}$ Hz (sampled at {fs} Hz)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)

plt.tight_layout()
plt.suptitle("FFT of Sampled Signals", fontsize=14, y=1.02)
plt.show()







import numpy as np
import matplotlib.pyplot as plt

# Sampling rate and frequency ranges
fs = 1000  # Sampling rate (Hz)
f_max = fs / 2  # Nyquist frequency (Hz)

# Analog frequency segments (in Hz)
analog_segments = {
    '1': (0, 250),
    '2': (250, 500),
    '3': (500, 750),
    '4': (750, 1000),
    '5': (1000, 1250),
    '6': (1250, 1500),
}

# Discrete frequency spectrum (in Hz)
discrete_segments = {
    'A': (-500, -333),
    'B': (-333, -167),
    'C': (-167, 0),
    'D': (0, 167),
    'E': (167, 333),
    'F': (333, 500),
}

# Generate the x-axis for plotting (frequency in Hz)
x = np.linspace(-f_max * 2, f_max * 2, 1000)

# Plot the analog spectrum, using vertical lines to represent the segments
plt.figure(figsize=(10, 6))

# Plot the analog spectrum segments
for segment, (start, end) in analog_segments.items():
    plt.axvline(x=start, color='black', linestyle='--', label=f'Analog Segment {segment} ({start}-{end} Hz)')
    plt.axvline(x=end, color='black', linestyle='--')

# Plot the discrete spectrum segments
for segment, (start, end) in discrete_segments.items():
    plt.fill_between(x, 0, 1, where=(x >= start) & (x <= end), alpha=0.2, label=f'Discrete Segment {segment} ({start}-{end} Hz)')

# Customize plot
plt.xlim(-f_max * 2, f_max * 2)
plt.ylim(0, 1)
plt.title('Analog and Discrete Spectrum with Aliasing')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.legend(loc='upper right')

plt.grid(True)
plt.show()







# Re-importing required libraries after code execution environment reset
import matplotlib.pyplot as plt
import numpy as np

# Define the sampling rate and signal frequency
fs = 1000  # Sampling frequency in Hz
f_signal = 700  # Signal frequency in Hz
n_replicas = 2  # Number of replicas on either side of the baseband

# Generate frequencies for aliases
frequencies = []
amplitudes = []
for k in range(-n_replicas, n_replicas + 1):
    f_pos = k * fs + f_signal
    f_neg = k * fs - f_signal
    frequencies.extend([f_pos, f_neg])
    amplitudes.extend([1, 1])  # All impulses have equal amplitude for a pure sine

# Plotting the spectrum
plt.figure(figsize=(10, 4))
plt.stem(frequencies, amplitudes)
plt.title("Spectrum of Sampled Signal x[n] = sin(2π700t)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.xticks(np.arange(-2000, 2001, 500))
plt.grid(True)
plt.xlim(-2000, 2000)
plt.ylim(0, 1.2)

plt.show()





import numpy as np
import matplotlib.pyplot as plt

# Parameters
fs = 56e6  # Sampling rate in Hz
fc = 70e6  # Center frequency of the original signal in Hz
bw = 14e6  # Bandwidth in Hz
f_min, f_max = -70e6, 70e6  # Plotting range

# Aliased center frequencies within range [-70 MHz, 70 MHz]
k_vals = range(-3, 4)  # Enough to cover the full range
aliases = []

for k in k_vals:
    alias_center = abs(fc - k * fs)
    if f_min <= alias_center <= f_max:
        aliases.append(alias_center)
    if f_min <= -alias_center <= f_max:
        aliases.append(-alias_center)

# Create frequency points for plotting each band (± BW/2 around each center)
bands = [(center - bw / 2, center + bw / 2) for center in aliases]

# Plotting
plt.figure(figsize=(12, 4))
for start, end in bands:
    plt.fill_between([start, end], 0, 1, alpha=0.4)

# Reference lines and formatting
plt.axhline(0, color='black', linewidth=0.5)
plt.title("Spectrum of Sampled Signal x(n) over [-70 MHz, 70 MHz]")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (normalized)")
plt.grid(True)
plt.xlim(f_min, f_max)
plt.ylim(0, 1.2)
plt.xticks(np.arange(f_min, f_max + 1, 14e6), rotation=45)

plt.tight_layout()
plt.show()





import numpy as np
import matplotlib.pyplot as plt

# Signal parameters
f_start = 2e3  # 2 kHz
f_end = 9e3    # 9 kHz
bandwidth = f_end - f_start

# Two sampling rates: one just right, one too low
fs_good = 18e3  # Minimum valid sampling rate
fs_bad = 12e3   # Too low: will cause overlap

# Generate spectrum replicas for both sampling scenarios
def generate_replicas(fs, num_replicas=3):
    centers = np.arange(-num_replicas, num_replicas + 1) * fs
    bands = [(center + f_start, center + f_end) for center in centers]
    return bands

bands_good = generate_replicas(fs_good)
bands_bad = generate_replicas(fs_bad)

# Plotting function
def plot_replicas(bands, fs, title, ax):
    for start, end in bands:
        ax.fill_between([start, end], 0, 1, alpha=0.5)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (normalized)")
    ax.grid(True)
    ax.set_xlim(-40e3, 40e3)
    ax.set_ylim(0, 1.2)
    ax.set_xticks(np.arange(-40e3, 41e3, fs))
    ax.axvline(-fs/2, color='red', linestyle='--', label='-fs/2')
    ax.axvline(fs/2, color='red', linestyle='--', label='fs/2')
    ax.legend()

# Plot both cases
fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
plot_replicas(bands_good, fs_good, "Spectrum with fs = 18 kHz (No Overlap)", axs[0])
plot_replicas(bands_bad, fs_bad, "Spectrum with fs = 12 kHz (Aliasing Occurs)", axs[1])

plt.tight_layout()
plt.show()







import numpy as np
import matplotlib.pyplot as plt

# Signal range
f_low = 247
f_high = 1175
bandwidth = f_high-f_low

# Minimum fs for bandpass sampling
fs_min = 2*bandwidth

# create replicas for ploting
def generate_bandpass_replicas(fs, num_replicas=3):
    centers = np.arange(-num_replicas, num_replicas+1) * fs
    #print(centers)
    bands = [(center+f_low, center+f_high) for center in centers]
    #print(bands)
    return bands

bands = generate_bandpass_replicas(fs_min)


# Plotting
plt.figure(figsize=(12,3))
for start, end in bands:
    plt.fill_between([start,end], 0, 1, alpha=0.4)

# Formatting
plt.axhline(0, color="black", linewidth=0.5)
plt.title(f"Soprano voice bandpass sampling spectrum (fs= {fs_min:.0f} Hz)")
plt.xlabel("Frequency Hz")
plt.ylabel("magnitude normalized")
plt.grid()
plt.xlim(-6000, 6000)
plt.ylim(0, 1.2)
plt.xticks(np.arange(-6000,6000+1, fs_min))
plt.axvline(-fs_min/2, color="red", linestyle='--', label='-fs/2')
plt.axvline(fs_min/2, color="red", linestyle='--', label='fs/2')
plt.legend()
plt.tight_layout()
plt.show()





import numpy as np
import matplotlib.pyplot as plt

# Parameters
f_signal = 15  # MHz, original center frequency of signal
fs = 12        # MHz, sample rate
target_fc = fs / 4  # Target center frequency after mixing

# Compute possible LO frequencies
f_LO_min = f_signal - target_fc
f_LO_max = f_signal + target_fc

# Frequencies for plotting
freqs = np.linspace(-20, 20, 1000)

# Generate spectra
def band_spectrum(center, width=2, label=None):
    start = center - width / 2
    end = center + width / 2
    plt.fill_between([start, end], 0, 1, alpha=0.5, label=label)

plt.figure(figsize=(14, 4))

# Plot original signal at f_signal
band_spectrum(f_signal, width=2, label="Original signal @ 15 MHz")

# Plot possible image spectrum locations after mixing
band_spectrum(target_fc, width=2, label="Target IF @ 3 MHz (fs/4)")
band_spectrum(-target_fc, width=2, label="Negative IF @ -3 MHz")

# Plot corresponding LO frequencies
plt.axvline(f_LO_min, color='red', linestyle='--', label=f'f_LO min = {f_LO_min:.1f} MHz')
plt.axvline(f_LO_max, color='green', linestyle='--', label=f'f_LO max = {f_LO_max:.1f} MHz')

# Formatting
plt.title("Mixer Operation and Spectral Centering at fs/4 (3 MHz)")
plt.xlabel("Frequency (MHz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.legend()
plt.xlim(-10, 25)
plt.ylim(0, 1.2)
plt.tight_layout()
plt.show()





import numpy as np
import matplotlib.pyplot as plt

# Parameters
B = 4e3  # 4 kHz bandwidth
fs = 16e3  # 16 kHz sampling rate
f_stop = fs - B  # Frequency where attenuation must be -60 dB

# Frequency axis (0 to 2*fs for showing replicas)
f = np.linspace(0, 2*fs, 1000)

# Ideal filter response
H = np.piecewise(f, [f <= B, (f > B) & (f < f_stop), f >= f_stop],
                 [0, lambda f: -60 * (f - B) / (f_stop - B), -60])

# Plotting
plt.figure(figsize=(12, 4))
plt.plot(f / 1e3, H, label='Anti-aliasing filter response')
plt.axvline(B / 1e3, color='green', linestyle='--', label='Bandwidth B = 4 kHz')
plt.axvline(f_stop / 1e3, color='red', linestyle='--', label='-60 dB point = fs - B = 12 kHz')

plt.title("Anti-Aliasing Filter Design: Attenuation Requirement")
plt.xlabel("Frequency (kHz)")
plt.ylabel("Attenuation (dB)")
plt.grid(True)
plt.legend()
plt.ylim(-70, 5)
plt.tight_layout()
plt.show()








import matplotlib.pyplot as plt
import numpy as np

# Given parameters
fc = 25_000  # Center frequency in Hz
B = 5_000    # Bandwidth in Hz

# Compute valid sampling frequency ranges for m = 5 to 15
m_values = np.arange(5, 16)
lower_bounds = (2 * fc - B) / m_values
upper_bounds = (2 * fc + B) / (m_values + 1)

# Plotting
plt.figure(figsize=(10, 6))
for i, m in enumerate(m_values):
    plt.plot([lower_bounds[i], upper_bounds[i]], [m, m], marker='|', color='blue', linewidth=4)

# Labels and formatting
plt.title("Valid Bandpass Sampling Frequency Ranges for Different m Values", fontsize=14)
plt.xlabel("Sampling Frequency $f_s$ (Hz)", fontsize=12)
plt.ylabel("m (Integer)", fontsize=12)
plt.grid(True)
plt.xlim(4000, 10000)
plt.ylim(4.5, 15.5)
plt.yticks(m_values)
plt.tight_layout()

plt.show()






import matplotlib.pyplot as plt
import numpy as np

# DFT size
N = 20

# Create an array of sample indices
m = np.arange(N)

# Create labels for the points
labels = [f'X({i})' for i in range(N)]

# Create colors: 
# Green for directly sent values, blue for conjugated (recoverable) values
colors = ['green' if (i <= 10) else 'blue' for i in m]

# Plot
plt.figure(figsize=(12, 2))
plt.scatter(m, np.zeros_like(m), c=colors, s=100)

# Annotate each point
for i in range(N):
    plt.text(m[i], 0.02, labels[i], ha='center', va='bottom', fontsize=9, rotation=45)

# Axis formatting
plt.yticks([])
plt.title('20-Point DFT: Values to Send vs Recover by Symmetry', fontsize=14)
plt.xlabel('DFT Index (m)', fontsize=12)
plt.axhline(0, color='black', linewidth=0.5)
plt.grid(False)
plt.xlim(-1, 20)

# Legend
plt.scatter([], [], c='green', label='Values to Send (Original)')
plt.scatter([], [], c='blue', label='Values Recovered by Symmetry')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()




import matplotlib.pyplot as plt
import numpy as np

# Sampling rate
fs = 1000  # Hz

# Range of N values to consider
N_values = np.arange(10, 60)  # from 10 to 59 samples

# Corresponding delta f values
delta_f = fs / N_values

# Target delta f
target_delta_f = 45  # Hz

# Plot
plt.figure(figsize=(10, 6))
plt.plot(N_values, delta_f, marker='o', label=r'$\Delta f = f_s / N$')
plt.axhline(target_delta_f, color='red', linestyle='--', label='Target 45 Hz')
plt.scatter(22, fs/22, color='green', s=100, label=r'Closest: $N=22$ ($\Delta f \approx 45.45$ Hz)')
plt.scatter(23, fs/23, color='purple', s=100, label=r'Closest: $N=23$ ($\Delta f \approx 43.48$ Hz)')

# Labels and title
plt.title('DFT Frequency Spacing vs. Number of Samples (N)', fontsize=14)
plt.xlabel('Number of DFT Points (N)', fontsize=12)
plt.ylabel('Frequency Spacing (Δf) [Hz]', fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()






import matplotlib.pyplot as plt
import numpy as np

# Define sample rate
fs = 44100  # Hz

# Define a range of N values
N_values = np.arange(1000, 50000, 1000)  # from 1000 to 49000 samples
delta_f_values = fs / N_values  # corresponding frequency spacings
duration_values = N_values / fs  # corresponding durations in seconds

# Create figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot delta_f vs N
ax1.plot(N_values, delta_f_values, marker='o')
ax1.axhline(1, color='red', linestyle='--', label='Target Δf = 1 Hz')
ax1.set_title('DFT Frequency Spacing vs. Number of Samples (N)', fontsize=14)
ax1.set_xlabel('Number of DFT Points (N)', fontsize=12)
ax1.set_ylabel('Frequency Spacing (Δf) [Hz]', fontsize=12)
ax1.grid(True)
ax1.legend()

# Plot duration vs N
ax2.plot(N_values, duration_values, marker='s', color='green')
ax2.axhline(1, color='purple', linestyle='--', label='Target Duration = 1 sec')
ax2.set_title('Signal Duration vs. Number of Samples (N)', fontsize=14)
ax2.set_xlabel('Number of DFT Points (N)', fontsize=12)
ax2.set_ylabel('Time Duration [seconds]', fontsize=12)
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()




import matplotlib.pyplot as plt
import numpy as np

# Parameters
fs = 3000  # Hz
fmax = fs / 2  # 1500 Hz
replication_spacing = fs  # 3000 Hz

# Frequency axis
f = np.linspace(-6000, 6000, 1000)  # frequency range from -6000 Hz to 6000 Hz

# Create a simple spectrum: nonzero between -fmax and fmax
X = np.zeros_like(f)
X[np.abs(f) <= fmax] = 1  # flat spectrum between -1500 Hz and 1500 Hz

# Create replications by shifting
X_rep1 = np.roll(X, int(replication_spacing / (f[1] - f[0])))
X_rep2 = np.roll(X, -int(replication_spacing / (f[1] - f[0])))

# Plot
plt.figure(figsize=(12, 6))
plt.plot(f, X, label='Original Spectrum', color='blue')
plt.plot(f, X_rep1, label='Replication +fs', color='green', linestyle='--')
plt.plot(f, X_rep2, label='Replication -fs', color='red', linestyle='--')

# Decorations
plt.title('Spectrum and Replications (Sampling at 3000 Hz)', fontsize=14)
plt.xlabel('Frequency [Hz]', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)
plt.grid(True)
plt.axvline(0, color='black', linestyle=':')
plt.legend()
plt.tight_layout()
plt.show()





import numpy as np
import matplotlib.pyplot as plt

# Parameters
fs = 22255  # Sampling frequency in Hz
N = 902     # Number of samples
f0 = 220    # Fundamental frequency (A3)
t = np.arange(N) / fs  # Time vector

# Create a signal with harmonics
harmonics = [1, 2, 3, 4, 5, 6]  # Harmonics to include
amplitudes = [1.0, 0.5, 0.3, 0.2, 0.1, 0.05]  # Decreasing amplitudes

x = np.zeros(N)
for i, amp in zip(harmonics, amplitudes):
    x += amp * np.sin(2 * np.pi * f0 * i * t)

# Compute DFT
X = np.fft.fft(x)
freqs = np.fft.fftfreq(N, d=1/fs)

# Only take the positive frequencies
half_N = N // 2
X_mag = np.abs(X[:half_N])
freqs = freqs[:half_N]

# Plot the result
plt.figure(figsize=(10, 6))
plt.plot(freqs, X_mag)
plt.title('Simulated Spectrum of a Perfect A3 Guitar Note')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.xlim(0, 2000)  # Show up to 2 kHz
plt.show()






import numpy as np
import matplotlib.pyplot as plt

# Parameters
fs = 22255  # Sampling frequency in Hz
N = 902     # Number of samples
f0 = 220    # Fundamental frequency (A3)
t = np.arange(N) / fs  # Time vector

# Create a signal with harmonics
harmonics = [1, 2, 3, 4, 5, 6]  # Harmonics to include
amplitudes = [1.0, 0.5, 0.3, 0.2, 0.1, 0.05]  # Decreasing amplitudes

x = np.zeros(N)
for i, amp in zip(harmonics, amplitudes):
    # Introduce slight frequency offset to cause leakage
    f_leaky = f0 * i * 1.02  # 2% offset to simulate non-bin alignment
    x += amp * np.sin(2 * np.pi * f_leaky * t)

# Compute DFT
X = np.fft.fft(x)
freqs = np.fft.fftfreq(N, d=1/fs)

# Only take the positive frequencies
half_N = N // 2
X_mag = np.abs(X[:half_N])
freqs = freqs[:half_N]

# Plot the result
plt.figure(figsize=(10, 6))
plt.plot(freqs, X_mag)
plt.title('Simulated Spectrum with Spectral Leakage (A3 Guitar Note)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.xlim(0, 2000)  # Show up to 2 kHz
plt.show()





import numpy as np
import matplotlib.pyplot as plt

# Function to create repeated Hanning windows
def repeated_hanning(K, N=16):
    h1 = 0.5 * (1 - np.cos(2 * np.pi * np.arange(N) / (N - 1)))
    return np.tile(h1, K)

# Parameters
N = 16
K2 = 2  # Two repetitions
K3 = 3  # Three repetitions

# Create sequences
h2 = repeated_hanning(K2, N)
h3 = repeated_hanning(K3, N)

# Compute DFTs
H2 = np.fft.fft(h2)
H3 = np.fft.fft(h3)

# DFT bin indices
m_H2 = np.arange(len(h2))
m_H3 = np.arange(len(h3))

# Only plot positive frequencies (up to half)
half_H2 = len(h2) // 2
half_H3 = len(h3) // 2

# Plotting
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.stem(m_H2[:half_H2], np.abs(H2[:half_H2]), basefmt=" ")
plt.title('|H2(m)| Spectrum (Two Repetitions)')
plt.xlabel('DFT Bin Index m')
plt.ylabel('Magnitude')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.stem(m_H3[:half_H3], np.abs(H3[:half_H3]), basefmt=" ")
plt.title('|H3(m)| Spectrum (Three Repetitions)')
plt.xlabel('DFT Bin Index m')
plt.ylabel('Magnitude')
plt.grid(True)

plt.tight_layout()
plt.show()







import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 64  # Number of points

# Time index
n = np.arange(N)

# Standard Hanning window (textbook formula)
h_standard = 0.5 * (1 - np.cos(2 * np.pi * n / (N - 1)))

# Alternate Hanning window (sin^2 formula)
h_alternate = np.sin(np.pi * n / (N - 1))**2

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(n, h_standard, label='Standard Hanning', linestyle='-', marker='o', markersize=4)
plt.plot(n, h_alternate, label='Alternate sin² Form', linestyle='--', marker='x', markersize=4)
plt.title('Comparison of Hanning Window Definitions')
plt.xlabel('Sample index n')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()

# Optional: Check numerical difference
error = np.max(np.abs(h_standard - h_alternate))
print(f'Maximum absolute difference between the two definitions: {error:.2e}')







import numpy as np
import matplotlib.pyplot as plt

# Original singal parameters
N = 32 # number of samples
f = 3 # frequency in cycles per N samples
Q = 128 # zero padded length Q > N, power of 2

# Time indices
n = np.arange(N)
n_zp = np.arange(Q)

# original signal cosine wave with 3 cycles over N samples
x = np.cos(2*np.pi*f*n/N)

# zero padded signal
x_zp = np.pad(x, (0, Q-N), 'constant')

# Compute DFTs
X_N = np.fft.fft(x, n=N)
X_Q = np.fft.fft(x_zp, n=Q)

# Frequency axes
freq_N = np.arange(N)
freq_Q = np.arange(Q) * N/Q # Scale to original N, more point but same min-max

# Plotting
plt.figure(figsize=(12,6))
plt.plot(freq_N, np.abs(X_N), 'o-', label='N-point DFT')
plt.plot(freq_Q, np.abs(X_Q), '--', label='Zero-padded Q-point DFT')
plt.title('Effect of Zero-Padding on the DFT')
plt.xlabel('Frequency Bin (scaled to N)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()




# Zoom in around the main peak
zoom_range = slice(25, 32)

plt.figure(figsize=(10, 5))
plt.plot(freq_N[zoom_range], np.abs(X_N)[zoom_range], 'o-', label='N-point DFT')
plt.plot(freq_Q[zoom_range.start * Q // N : zoom_range.stop * Q // N],
         np.abs(X_Q)[zoom_range.start * Q // N : zoom_range.stop * Q // N],
         '--', label='Zero-padded Q-point DFT')
plt.title('Zoomed-in View: Effect of Zero-Padding on the DFT')
plt.xlabel('Frequency Bin (scaled to N)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()






# Apply a Hanning window to the original signal
hanning_window = np.hanning(N)
x_windowed = x * hanning_window

# Zero-pad the windowed signal to length Q
xw_zp = np.pad(x_windowed, (0, Q - N), 'constant')

# Compute DFT of the windowed and zero-padded signal
Xw_Q = np.fft.fft(xw_zp, n=Q)

# Plot the zoomed-in spectrum of the windowed signal
plt.figure(figsize=(10, 5))
plt.plot(freq_Q[10 * Q // N : 30 * Q // N],
         np.abs(Xw_Q)[10 * Q // N : 30 * Q // N],
         label='Zero-padded Q-point DFT (Hanning window)')
plt.title('Zoomed-in Spectrum Using Hanning Window')
plt.xlabel('Frequency Bin (scaled to N)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()







import numpy as np
import matplotlib.pyplot as plt

# Define original parameters
N = 64  # Original number of samples
fs = 100 * N  # Sampling frequency from given spacing of 100 Hz
delta_f_original = fs / N

# Zero-padding
M = 5 * N  # New DFT length
delta_f_padded = fs / M


# Frequency axes for original and zero-padded DFTs
freq_axis_N = np.fft.fftfreq(N, d=1/fs)
freq_axis_M = np.fft.fftfreq(M, d=1/fs)

# Example signal: a cosine at bin 5 of N-point DFT
x = np.cos(2 * np.pi * 5 * np.arange(N) / N)
x_zp = np.pad(x, (0, M - N), 'constant')  # zero-padded signal

# DFTs
X_N = np.fft.fft(x, n=N)
X_M = np.fft.fft(x_zp, n=M)


# Plot the magnitude spectra (first half only)
plt.figure(figsize=(12, 6))
plt.plot(freq_axis_N[:N//2], np.abs(X_N[:N//2]), 'o-', label='Original DFT (Δf = 100 Hz)')
plt.plot(freq_axis_M[:M//2], np.abs(X_M[:M//2]), '--', label='Zero-padded DFT (Δf = 20 Hz)')
plt.title('Effect of Zero-Padding on DFT Frequency Resolution')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
