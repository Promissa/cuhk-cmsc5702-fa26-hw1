import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(
    description="Plot a temperature grid from a text file."
)
parser.add_argument("filename", help="Path to the temperature data file")
args = parser.parse_args()

input_file = Path(args.filename)
output_file = input_file.with_suffix(".png")

temperature = np.loadtxt(input_file, ndmin=2)

fig, ax = plt.subplots(figsize=(8, 6))

image = ax.imshow(
    temperature,
    cmap="coolwarm",  # Blue: lower temperature; red: higher temperature
    origin="upper",
    interpolation="nearest",
    aspect="equal",
)

fig.colorbar(image, ax=ax, label="Temperature")

ax.set_xlabel("Column index")
ax.set_ylabel("Row index")
ax.set_title(f"Final temperature distribution — {input_file.name}")

fig.tight_layout()
fig.savefig(output_file, dpi=300)
plt.show()
