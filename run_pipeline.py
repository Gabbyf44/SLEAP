import tkinter as tk
from pathlib import Path
from tkfilebrowser import askopendirnames

from dataset import make_expt_dataset
from perframe.create_perframe import export_perframe
from perframe.trx_mat import save_trx
from params import BASE_PATH

def main():
    """
    Entry point for batch extraction of features from SLEAP analysis files.

    Opens a GUI to select experiment folders, locates a single '*.analysis.h5'
    file in each folder, and generates a corresponding '*.features.h5' file
    using `make_expt_dataset`.

    Handles only user interaction, input validation, and per-experiment
    orchestration. Assumes fixed, proofread identities in the input files.

    Creates one output file per experiment and reports progress to stdout.

    Returns:
        None
    """
    root = tk.Tk()
    root.withdraw()

    expt_folders = askopendirnames(
        title="Select one or more experiment folders",
        initialdir=BASE_PATH
    )

    if not expt_folders:
        print("No experiment folders selected")
        return

    for expt_folder in expt_folders:
        expt_folder = expt_folder.strip()
        print(f"\nProcessing experiment folder: {expt_folder}")

        # Decide which inference file exists
        analysis_files = list(Path(expt_folder).glob("*.analysis.h5"))

        if len(analysis_files) == 0:
            print("\tNo inference H5 found, skipping")
            continue
        elif len(analysis_files) > 1:
            print("\tMultiple analysis H5 files found, skipping")
            continue

        analysis_path = str(analysis_files[0])

        features_path = analysis_path.replace(".analysis.h5", ".features.h5")

        try:
            make_expt_dataset(
                expt_folder,
                h5_file=analysis_path,
                output_path=features_path,
                overwrite=True
            )
        except Exception as e:
            print(f"\tERROR processing {expt_folder}: {e}")

        # Decide which features file exists
        features_files = list(Path(expt_folder).glob("*.features.h5"))

        if len(features_files) == 0:
            print("\tNo features H5 found, skipping")
            continue
        elif len(features_files) > 1:
            print("\tMultiple features H5 files found, skipping")
            continue

        features_h5 = features_files[0]
        perframe_dir = Path(expt_folder) / "perframe"

        try:
            print(f"\tUsing features file: {features_h5.name}")
            print(f"\tCreating perframe directory: {perframe_dir}")

            export_perframe(
                features_h5=features_h5,
                perframe_dir=perframe_dir,
                overwrite=True,
            )

            trx_path = Path(expt_folder) / "trx.mat"
            save_trx(features_h5, trx_path, timestamps=None, overwrite=True)

        except Exception as e:
            print(f"\tERROR processing {expt_folder}: {e}")

    print("\nDone.")

if __name__ == "__main__":
    main()