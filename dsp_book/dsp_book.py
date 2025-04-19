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
