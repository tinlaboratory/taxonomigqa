import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def autopct_for_big_slices(pct):
    """
    Show label only if the slice is >= 2%.
    Otherwise, return an empty string.
    """
    return f"{pct:.1f}%" if pct >= 2 else ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file", default="./single_nouns.json", help="JSON file with data"
    )
    args = parser.parse_args()

    data = json.load(open(args.file))

    types = []
    for key, val in data.items():
        types.append(val["types"]["detailed"])
    df = pd.DataFrame(types)

    freq_counts = df.value_counts(normalize=True)

    labels = []
    for row in freq_counts.items():
        if row[1] > 0.02:
            labels.append(f"{row[0][0]} - {row[1] * 100:.1f}%")

    freq_counts.plot(
        kind="pie",
        autopct=autopct_for_big_slices,  # Use our custom function
        startangle=140,
        labels=None,  # Remove default slice labels
    )

    plt.legend(labels, title="Categories", loc="center left", bbox_to_anchor=(1.0, 0.5))
    plt.title("Category Distribution")
    plt.ylabel("")  # remove y-label for a cleaner look
    plt.show()
