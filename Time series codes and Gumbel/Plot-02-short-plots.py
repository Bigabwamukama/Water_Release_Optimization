import pandas as pd
import matplotlib.pyplot as plt

# Set global font sizes and styles
plt.rcParams.update({
    "font.size": 18,   # General font size
    "axes.labelsize": 22,  # Axis labels
    "xtick.labelsize": 16,  # X-axis ticks
    "ytick.labelsize": 16,  # Y-axis ticks
    "legend.fontsize": 18,  # Legend size
    "axes.linewidth": 2,  # Border thickness
    "xtick.major.size": 8,  # Major tick size
    "ytick.major.size": 8
})

# Load the Excel file
file_path = "d:/PYTHON/FYP data plots/FYP Data sheets.xlsx"  
xls = pd.ExcelFile(file_path)

# Load the "Short plots" sheet, ensuring the first row is used as headers
df_short = xls.parse("Short plots", header=0)

# Rename columns properly
df_short = df_short.iloc[:, :3]  # Keep only the first three columns
df_short.columns = ["Year", "Max_Discharge", "Max_Lake_Level"]

# Convert columns to numeric, handling any non-numeric values
df_short["Year"] = pd.to_numeric(df_short["Year"], errors="coerce")
df_short["Max_Discharge"] = pd.to_numeric(df_short["Max_Discharge"], errors="coerce")
df_short["Max_Lake_Level"] = pd.to_numeric(df_short["Max_Lake_Level"], errors="coerce")

# Drop any rows with missing values
df_short.dropna(inplace=True)

### First Plot: Max_Lake_Level vs Year (Blue for water levels)
plt.figure(figsize=(12, 7))
plt.plot(df_short["Year"], df_short["Max_Lake_Level"], linestyle="-", 
         color="blue", linewidth=1.5)
plt.xlabel("Time (Years)")
plt.ylabel("Annual Maximum Water Level (m)")

# Show all years on x-axis
years = df_short["Year"].unique()
plt.xticks(years, rotation=45)

plt.xlim(df_short["Year"].min(), 2025)
plt.grid(False)  # Removed grid
plt.tight_layout()
plt.show()

### Second Plot: Max_Discharge vs Year (Purple for discharge)
plt.figure(figsize=(12, 7))
plt.plot(df_short["Year"], df_short["Max_Discharge"], linestyle="-", 
         color="purple", linewidth=1.5)
plt.xlabel("Time (Years)")
plt.ylabel("Annual Maximum Discharge (cumecs)")

# Show all years on x-axis
plt.xticks(years, rotation=45)

plt.xlim(df_short["Year"].min(), 2025)
plt.grid(False)  # Removed grid
plt.tight_layout()
plt.show()

### Third Plot: Combined Max_Discharge & Max_Lake_Level vs Year
fig, ax1 = plt.subplots(figsize=(12, 7))

# Plot Max_Lake_Level (Blue, left y-axis)
ax1.plot(df_short["Year"], df_short["Max_Lake_Level"], linestyle="-", 
         color="blue", linewidth=1.5, label="Annual Max Water Level")
ax1.set_xlabel("Time (Years)")
ax1.set_ylabel("Annual Maximum Water Level (m)", color="blue")
ax1.tick_params(axis="y", labelcolor="blue")

# Second y-axis for Max_Discharge (Purple)
ax2 = ax1.twinx()
ax2.plot(df_short["Year"], df_short["Max_Discharge"], linestyle="-", 
         color="purple", linewidth=1.5, label="Annual Max Discharge")
ax2.set_ylabel("Annual Maximum Discharge (cumecs)", color="purple")
ax2.tick_params(axis="y", labelcolor="purple")

# Show all years on x-axis
ax1.set_xticks(years)
ax1.set_xticklabels(years, rotation=45)

# Set x-axis limits
ax1.set_xlim(df_short["Year"].min(), 2025)

plt.grid(False)  # Removed grid

# Adding a combined legend inside the plot area at the upper left
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
fig.legend(lines1 + lines2, labels1 + labels2, loc="upper left", bbox_to_anchor=(0.1, 0.9))

plt.tight_layout()
plt.show()
