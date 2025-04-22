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


