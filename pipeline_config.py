"""
Shared paths for the Prop_hull surrogate pipeline.

Change INFILL_ROUND (and optionally INITIAL_TAG) when starting a new iteration.
All scripts import paths from here instead of hard-coding filenames.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "New_training"

# --- change these between iterations ---
INFILL_ROUND = 10
INITIAL_TAG = "initial_sampling_new"

# All New_training tables store design-vars (no con-point files in this campaign).
CONPOINT_TRAINING_NAMES = frozenset()


def training_data_files():
    """Initial training file plus infill files in numeric round order (only existing paths)."""
    paths = [DATA_DIR / "training_data_initial.dat"]

    def infill_round_num(path):
        # training_data_infill12.dat -> 12  (numeric, not lexicographic)
        stem = path.stem  # training_data_infill12
        suffix = stem.replace("training_data_infill", "", 1)
        return int(suffix)

    infill_paths_list = [
        path for path in DATA_DIR.glob("training_data_infill*.dat") if path.is_file()
    ]
    infill_paths_list.sort(key=infill_round_num)
    paths.extend(infill_paths_list)
    return [path for path in paths if path.is_file()]


def initial_sampling_paths():
    """Output paths for Initial_sampling.py."""
    geometry_dir = ROOT / "geometry" / INITIAL_TAG
    return {
        "existing_input": ROOT / f"rejected_{INITIAL_TAG}_control_points.txt",
        "generated": ROOT / f"generated_{INITIAL_TAG}_control_points.txt",
        "accepted": ROOT / f"input__var_values_{INITIAL_TAG}.dat",
        "rejected": ROOT / f"rejected_{INITIAL_TAG}_control_points.txt",
        "plot": ROOT / f"{INITIAL_TAG}_samples.png",
        "geometry_dir": geometry_dir,
    }


def cad_output_paths(case_id, geometry_dir=None):
    """Per-case geometry files (ORCA coordinates, IGES, optional STL)."""
    case_id = int(case_id)
    folder = Path(geometry_dir) if geometry_dir is not None else ROOT / "geometry" / INITIAL_TAG
    return {
        "dir": folder,
        "orca_dat": folder / f"ORCA{case_id}.dat",
        "iges": folder / f"sample_blade{case_id}.iges",
        "stl": folder / f"sample_blade{case_id}.stl",
    }


def infill_paths(round_num=None):
    """Standard filenames for one round (suggest → CFD → training).

    Round 0 is the initial sampling round and is tagged 'initial' so it is not
    confused with the actual infill rounds (1, 2, ...) that follow.
    """
    r = INFILL_ROUND if round_num is None else int(round_num)
    tag = "initial" if r == 0 else f"infill{r}"
    return {
        "control_points": DATA_DIR / f"{tag}_control_points.txt",
        "control_points_con": DATA_DIR / f"{tag}_control_points_con_points.txt",
        "predictions": DATA_DIR / f"{tag}_predictions.txt",
        "results": DATA_DIR / f"Results_{tag}.txt",
        "poly_fit": DATA_DIR / f"poly_fit_eq_{tag}.txt",
        "training_data": DATA_DIR / f"training_data_{tag}.dat",
        "plot_dir": DATA_DIR / f"intersection_plots_{tag}",
        "geometry_dir": ROOT / "geometry" / tag,
        "cad_accepted": DATA_DIR / f"{tag}_cad_accepted.txt",
        "cad_rejected": DATA_DIR / f"{tag}_cad_rejected.txt",
        "cad_plot": DATA_DIR / f"{tag}_cad_samples.png",
        # Control points to map results back to design vars for training.
        # Initial round: row index == case id (no case-id column).
        # Infill rounds: cad_accepted carries an explicit case-id column
        # (case ids are offset, e.g. 50..67), so use that mapping.
        "training_control_points": (
            DATA_DIR / f"{tag}_control_points.txt"
            if r == 0
            else DATA_DIR / f"{tag}_cad_accepted.txt"
        ),
    }
