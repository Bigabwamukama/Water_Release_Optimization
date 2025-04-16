import pandas as pd
import matplotlib.pyplot as plt

# Load the Excel file
file_path = "d:/PYTHON/FYP data plots/FYP Data sheets.xlsx"  # Update the path if needed
xls = pd.ExcelFile(file_path)

# Load the "Return period plot for Gumbel" sheet
df_gumbel = xls.parse("Return period plot for Gumbel", header=0)

# Convert columns to numeric
df_gumbel["Return Period"] = pd.to_numeric(df_gumbel["Return Period"], errors="coerce")
df_gumbel["Return Periodss"] = pd.to_numeric(df_gumbel["Return Periodss"], errors="coerce")
df_gumbel["Observed Water Level"] = pd.to_numeric(df_gumbel["Observed Water Level"], errors="coerce")
df_gumbel["Predicted Water Level"] = pd.to_numeric(df_gumbel["Predicted Water Level"], errors="coerce")

# Plot the data
plt.figure(figsize=(10, 5))

# Increase font sizes for tick labels
plt.tick_params(axis='both', which='major', labelsize=14)  # Increased tick label size

# Plot Observed Water Level using "Return Period"
plt.plot(df_gumbel["Return Period"], df_gumbel["Observed Water Level"], 
         marker="o", markersize=4, linestyle="--", color="blue", label="Observed Water Level")

# Plot Predicted Water Level using "Return Periodss"
plt.plot(df_gumbel["Return Periodss"], df_gumbel["Predicted Water Level"], 
         marker="o", markersize=4, linestyle="-", color="orange", label="Gumbel Predicted water levels")

# Customize axis limits and labels with larger font sizes
plt.xlim(0, 200)  # Extend x-axis to 200
plt.ylim(10, 15)  # Set y-axis range to match the graph in the image
plt.xlabel("Return Period (Years)", fontsize=16)  # Increased from 12
plt.ylabel("Annual Maximum Water Level (m)", fontsize=16)  # Increased from 12
plt.legend(loc="lower right", fontsize=14)  # Increased from 10

# Add grid lines for better visualization
plt.grid(visible=True, linestyle="--", alpha=0.7)

# Show the plot
plt.show()