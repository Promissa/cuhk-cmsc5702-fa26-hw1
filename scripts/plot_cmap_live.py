import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def main():
    parser = argparse.ArgumentParser(
        description="Display a live temperature color map from a text file."
    )
    parser.add_argument(
        "filename",
        help="Path to the whitespace-separated temperature data file",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=1000,
        help="Refresh interval in milliseconds (default: 1000)",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=0.0,
        help="Minimum temperature for the color scale (default: 0)",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=1000.0,
        help="Maximum temperature for the color scale (default: 1000)",
    )
    args = parser.parse_args()

    if args.interval <= 0:
        parser.error("--interval must be positive.")
    if not np.isfinite([args.vmin, args.vmax]).all():
        parser.error("--vmin and --vmax must be finite.")
    if args.vmin >= args.vmax:
        parser.error("--vmin must be smaller than --vmax.")

    # The file must contain a valid grid when the script starts.
    try:
        temperature = np.loadtxt(args.filename, ndmin=2)
    except (OSError, ValueError) as exc:
        parser.error(f"Cannot load {args.filename!r}: {exc}")

    if temperature.size == 0 or not np.isfinite(temperature).all():
        parser.error("The initial grid must be nonempty and contain finite values.")

    expected_shape = temperature.shape

    fig, ax = plt.subplots(figsize=(8, 6))

    image = ax.imshow(
        temperature,
        cmap="coolwarm",  # Blue: low temperature; red: high temperature
        origin="upper",
        interpolation="nearest",
        aspect="equal",
        vmin=args.vmin,
        vmax=args.vmax,
    )

    fig.colorbar(image, ax=ax, label="Temperature")
    ax.set_xlabel("Column index")
    ax.set_ylabel("Row index")
    ax.set_title(f"Live temperature distribution\n{args.filename}")
    fig.tight_layout()

    def update(frame):
        try:
            latest = np.loadtxt(args.filename, ndmin=2)

            # Keep the previous image if a snapshot appears invalid.
            if latest.shape != expected_shape:
                return (image,)
            if not np.isfinite(latest).all():
                return (image,)

            image.set_data(latest)

        except (OSError, ValueError):
            # The writer may temporarily be rewriting the file.
            pass

        return (image,)

    # Retain this reference while the plotting window is open.
    animation = FuncAnimation(
        fig,
        update,
        interval=args.interval,
        blit=False,
        cache_frame_data=False,
    )

    plt.show()


if __name__ == "__main__":
    main()
