import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file.
# If the CSV file has an extra trailing comma per row, we can ignore extra unnamed columns.
data = pd.read_csv("Lab-3/CombinedPrecisionResults.csv")
# Optionally drop any unnamed columns that resulted from trailing commas.
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

# Clean column names (if needed)
data.columns = data.columns.str.strip()

# Convert Trial and Value columns to numeric (in case they are read as strings)
data['Trial'] = pd.to_numeric(data['Trial'], errors='coerce')
data['Value'] = pd.to_numeric(data['Value'], errors='coerce')

# Get all unique precision types
precision_types = data['PrecisonType'].unique()

plt.figure(figsize=(10, 6))

# Process and plot data for each precision type
for pt in precision_types:
    # Select rows for this precision type and sort by trial number
    subset = data[data['PrecisonType'] == pt].sort_values(by='Trial')
    # Compute the cumulative maximum (best accuracy so far) for each trial.
    best_so_far = subset['Value'].cummax()
    # Plot the cumulative best accuracy vs. trial.
    plt.plot(subset['Trial'], best_so_far, marker='o', label=pt)

plt.xlabel('Trial')
plt.ylabel('Best Accuracy So Far')
plt.title('Cumulative Best Accuracy by Trial for Each Precision Type')
plt.legend(title="Precision Type")

plt.grid(True)
plt.tight_layout()
plt.show()
