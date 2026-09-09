import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================
# read data
# =========================
file_path = r"C:\Users\13709\Desktop\Draw_MS2.xlsx"
df = pd.read_excel(file_path)

plt.rcParams["font.family"] = "Arial"

# =========================
# function
# =========================
def parse_array(x):
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return x
    return [
        float(i) for i in str(x)
        .replace("[", "")
        .replace("]", "")
        .split(",")
        if i != ""
    ]

plt.style.use("seaborn-v0_8-white")

# =========================
# plotting
# =========================
for idx, row in df.iterrows():

    mz1 = np.array(parse_array(row["MS2_mz"]))
    int1 = np.array(parse_array(row["MS2_Intensity"]))

    mz2 = np.array(parse_array(row["MS2_mz2"]))
    int2 = np.array(parse_array(row["MS2_Intensity2"]))

    if len(mz1) == 0 or len(mz2) == 0:
        continue

    # =========================
    # normalize (0-100)
    # =========================
    int1 = int1 / np.max(int1) * 100
    int2 = int2 / np.max(int2) * 100

    # =========================
    # figure
    # =========================
    fig, ax = plt.subplots(figsize=(6, 3))

    # Sample
    ax.vlines(
        mz1,
        0,
        int1,
        color="black",
        linewidth=1.5,
        alpha=0.9,
        label="Sample"
    )

    # Standard
    ax.vlines(
        mz2,
        0,
        -int2,
        color="#D62728",
        linewidth=1.5,
        alpha=0.9,
        label="Standard"
    )

    # baseline
    ax.axhline(
        0,
        color="gray",
        linewidth=1
    )

    all_mz = np.concatenate([mz1, mz2])

    xmin = np.floor(np.min(all_mz) / 50) * 50
    xmax = np.ceil(np.max(all_mz) / 50) * 50

    ax.set_xlim(
        xmin,
        xmax
    )

    max_int = max(
        np.max(int1),
        np.max(int2)
    )

    ax.set_ylim(
        -110,
        110
    )

    ax.set_xticks(
        [50, 150, 250, 350, 450]
    )

    ax.set_yticks(
        [-100, -50, 0, 50, 100]
    )

    ax.set_yticklabels(
        ["100", "50", "0", "50", "100"]
    )

    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=4,
        width=1,
        labelsize=10,
        colors="black"
    )

    ax.tick_params(
        axis="x",
        which="major",
        labelbottom=False
    )

    ax.tick_params(
        axis="y",
        which="major",
        labelleft=False
    )

    # black label
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)

    plt.tight_layout()

    plt.savefig(
        f"C:\\Users\\13709\\Desktop\\MS2_mirror_{idx}.png",
        dpi=300,
        transparent=True,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()