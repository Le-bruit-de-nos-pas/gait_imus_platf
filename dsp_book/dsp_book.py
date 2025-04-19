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

