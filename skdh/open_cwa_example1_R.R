
library(read.cwa)
file = system.file("extdata", "ax3_testfile.cwa.gz", package = "read.cwa")
out = read_cwa(file)

file = system.file("extdata", "ax3_testfile.cwa.gz", package = "read.cwa")

out = read_cwa(file)

head(out)

out = read_cwa(file, xyz_only = TRUE)


data.table::fwrite(out$data, "out.csv")

out = out$data

library(tidyverse)

# Step 1: Calculate the magnitude of acceleration
out <- out %>%
  mutate(accel_mag = sqrt(X^2 + Y^2 + Z^2))

# Step 2: Compute velocity by integrating acceleration over time
# Convert time to seconds
out <- out %>%
  arrange(time) %>%  # Ensure data is ordered
  mutate(time_diff = as.numeric(difftime(time, lag(time, default = first(time)), units = "secs")),
         speed = cumsum(accel_mag * time_diff)) # Integrate acceleration

# Step 3: Plot speed over time
ggplot(out, aes(x = time, y = speed)) +
  geom_line(color = "blue") +
  labs(title = "Speed over Time",
       x = "Time",
       y = "Speed (m/s)") +
  theme_minimal()





library(dplyr)
library(lubridate)
library(signal)  # For filtering

# Set parameters
sampling_rate <- 100  # 100 Hz
dt <- 1 / sampling_rate  # Time step (0.01s)
duration <- 5 * 60    # 5 minutes
num_samples <- sampling_rate * duration

# Generate timestamps
time_seq <- seq(ymd_hms("2024-03-11 12:00:00"), by = dt, length.out = num_samples)

# Simulate realistic acceleration values (walking/running)
set.seed(42)
X <- sin(seq(0, 20 * pi, length.out = num_samples)) + rnorm(num_samples, 0, 0.2)
Y <- cos(seq(0, 20 * pi, length.out = num_samples)) + rnorm(num_samples, 0, 0.2)
Z <- rep(9.81, num_samples) + rnorm(num_samples, 0, 0.3)  # Gravity + noise

# Remove gravity from Z-axis
Z_corrected <- Z - 9.81

# Remove mean drift from acceleration
X_detrended <- X - mean(X)
Y_detrended <- Y - mean(Y)
Z_detrended <- Z_corrected - mean(Z_corrected)

# Apply high-pass filter to remove low-frequency drift
hp_cutoff <- 0.1 / (sampling_rate / 2)  # Normalized 0.1 Hz cutoff
butter_highpass <- butter(2, hp_cutoff, type = "high")
X_filtered <- filtfilt(butter_highpass, X_detrended)
Y_filtered <- filtfilt(butter_highpass, Y_detrended)
Z_filtered <- filtfilt(butter_highpass, Z_detrended)

# Compute total acceleration magnitude (without gravity)
accel_mag <- sqrt(X_filtered^2 + Y_filtered^2 + Z_filtered^2)

# Integrate acceleration to get velocity
velocity <- cumsum(accel_mag * dt)

# Apply zero-velocity update (ZUPT): reset speed when acceleration is near zero
velocity[accel_mag < 0.1] <- 0  # If acceleration is very low, assume no motion

# Apply low-pass filter to smooth velocity
lp_cutoff <- 2 / (sampling_rate / 2)  # Normalized 2 Hz cutoff
butter_lowpass <- butter(2, lp_cutoff, type = "low")
velocity_filtered <- filtfilt(butter_lowpass, velocity)

# Create a data frame
accel_data <- tibble(time = time_seq, X = X_filtered, Y = Y_filtered, Z = Z_filtered, 
                     accel_mag = accel_mag, speed = velocity_filtered)

# Plot speed over time
ggplot2::ggplot(accel_data, aes(x = time, y = speed)) +
  geom_line(color = "blue") +
  labs(title = "Realistic Human Speed Over Time", x = "Time", y = "Speed (m/s)") +
  theme_minimal()

plot(accel_data$time, accel_data$speed)

# Show first few rows
print(accel_data, n = 10)
