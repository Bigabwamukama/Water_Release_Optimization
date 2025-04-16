import pandas as pd
import matplotlib.pyplot as plt

# Set font sizes and styles to resemble the reference plot
plt.rcParams.update({
    "font.size": 18,   # General font size similar to reference
    "axes.labelsize": 22,  # Axis labels size
    "xtick.labelsize": 18,  # X-axis tick size
    "ytick.labelsize": 18,  # Y-axis tick size
    "legend.fontsize": 18,  # Legend font size
    "axes.linewidth": 2,  # Border thickness
    "xtick.major.size": 8,  # Major tick size
    "ytick.major.size": 8
})

# Load the Excel file
file_path = "d:/PYTHON/FYP data plots/FYP Data sheets.xlsx" #repalce the file path appropriately
xls = pd.ExcelFile(file_path)

# Load the "Observed AMWLs" sheet, skipping the first row (headers in the second row)
df = xls.parse("Observed AMWLs", skiprows=1)

# Rename columns
df = df.iloc[:, :2]  # Keep only the first two columns
df.columns = ["Year", "Water Level"]

# Convert columns to numeric
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["Water Level"] = pd.to_numeric(df["Water Level"], errors="coerce")

# Drop rows with missing values
df.dropna(inplace=True)

# Plot the time series
plt.figure(figsize=(10, 7))  # Adjusted to match reference proportions
plt.plot(df["Year"], df["Water Level"], marker="", linestyle="-", color="blue", linewidth=2.5, markersize=5, alpha=0.7)

# Customize the axis labels
plt.xlabel("Time (Years)")
plt.ylabel("Annual Maximum Water Level (m)")

# Adjust x-axis limits
plt.xlim(df["Year"].min(), 2025)

# Add grid with light styling
plt.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)

# Show the plot
plt.show()
