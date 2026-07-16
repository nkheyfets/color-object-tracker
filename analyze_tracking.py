import csv

import matplotlib.pyplot as plt


def load_tracking_data(filename: str) -> tuple[list[int], list[int]]:
    x_positions = []
    y_positions = []

    with open(filename, "r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            x_positions.append(int(row["Center X"]))
            y_positions.append(int(row["Center Y"]))

    return x_positions, y_positions


def main() -> None:
    x_positions, y_positions = load_tracking_data(
        "tracking_data.csv"
    )

    if not x_positions:
        print("No tracking data was found.")
        return

    plt.figure(figsize=(8, 6))
    plt.plot(
        x_positions,
        y_positions,
        marker="o",
        markersize=2,
    )

    plt.title("Tracked Object Movement")
    plt.xlabel("Horizontal Position (pixels)")
    plt.ylabel("Vertical Position (pixels)")

    # Image coordinates increase downward, so reverse the y-axis.
    plt.gca().invert_yaxis()

    plt.grid(True)
    plt.tight_layout()

    output_file = "screenshots/tracking_movement_graph.png"
    plt.savefig(output_file, dpi=300)

    print(f"Graph saved as {output_file}")

    plt.show()

if __name__ == "__main__":
    main()