import argparse
import sys
import shutil
import subprocess
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Define base paths relative to this script's location (tests/cpp/)
SCRIPT_DIR = Path(__file__).parent
BASE_TEST_DIR = SCRIPT_DIR
DATA_DIR = BASE_TEST_DIR / "data"
SCRIPTS_DIR = BASE_TEST_DIR / "scripts"


def find_available_suites():
    """Finds available generation scripts and extracts suite names."""
    available_suites = {}
    if not SCRIPTS_DIR.is_dir():
        logging.warning(f"Scripts directory not found: {SCRIPTS_DIR}")
        return available_suites

    script_files = sorted(
        list(SCRIPTS_DIR.glob("generate_*_data.py"))
    )  # Sort for consistent order
    for script_path in script_files:
        parts = script_path.stem.split("_")
        if len(parts) >= 3 and parts[0] == "generate" and parts[-1] == "data":
            suite_name = "_".join(parts[1:-1])
            if suite_name:
                available_suites[suite_name] = script_path
            else:
                logging.warning(
                    f"Could not determine suite name from script: {script_path.name}"
                )
        else:
            logging.warning(
                f"Script name {script_path.name} does not match expected pattern 'generate_<suite>_data.py'"
            )
    return available_suites


def run_script(script_path: Path):
    """Executes a given python script."""
    script_name = script_path.name
    logging.info(f"Running script: {script_name}...")
    try:
        result = subprocess.run(
            [sys.executable, str(script_path.resolve())],
            check=True,
            cwd=SCRIPTS_DIR,
            capture_output=True,
            text=True,
        )
        logging.info(f"Finished script: {script_name}")
        if result.stdout:
            logging.info(f"--- {script_name} STDOUT ---\n{result.stdout.strip()}")
        if result.stderr:
            logging.warning(f"--- {script_name} STDERR ---\n{result.stderr.strip()}")

    except FileNotFoundError:
        logging.error(
            f"Error: Python interpreter not found or script missing: {script_path}"
        )
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running script: {script_name}")
        logging.error(f"Return code: {e.returncode}")
        if e.stdout:
            logging.error(f"--- {script_name} STDOUT ---\n{e.stdout.strip()}")
        if e.stderr:
            logging.error(f"--- {script_name} STDERR ---\n{e.stderr.strip()}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred while running {script_name}: {e}")
        sys.exit(1)


def main():
    available_suites = find_available_suites()
    available_suites_str = (
        f"Available suites: {', '.join(sorted(available_suites.keys()))}"
        if available_suites
        else "No suites found."
    )

    # --- Argument Parser Setup ---
    parser = argparse.ArgumentParser(
        description=(
            "Regenerates C++ test reference data by running scripts in the 'scripts/' directory.\n"
            "WARNING: This script deletes data before regenerating.\n"
            "Default behavior (no arguments): Prints usage hint, then prompts before deleting the entire 'data/' directory and regenerating all suites.\n"
            "Specify suite names: Deletes only specified suite data directories and regenerates only those suites (no prompt).\n"
            "Use --force/-f: Skips the confirmation prompt when regenerating all suites."
        ),
        epilog=available_suites_str,
        formatter_class=argparse.RawDescriptionHelpFormatter,  # Preserve newline formatting in description
    )
    parser.add_argument(
        "suites",
        metavar="SUITE",
        nargs="*",  # Zero or more suite names
        help="Optional: Specific test suite(s) to regenerate (e.g., fft window cqt).",
    )
    parser.add_argument(
        "--force",
        "-f",  # Added short flag alias
        action="store_true",
        help="Force regeneration of ALL data without confirmation. Only used when no specific suites are provided.",
    )

    # --- Initial Hint for No Arguments ---
    # If run as just "python data.py", print usage hint first.
    # The script will then continue and enter the confirmation mode below.
    if len(sys.argv) == 1:
        print(
            "Hint: Regenerating all suites. Provide specific suite names to regenerate only those."
        )
        print("      Use --help for detailed instructions and available suites.")
        print("---")  # Separator before potential prompt

    # --- Parse Arguments ---
    args = parser.parse_args()

    # --- Validate Script Availability ---
    if not available_suites:
        logging.error(
            "No generation scripts found in 'scripts/' directory matching 'generate_<suite>_data.py' pattern."
        )
        sys.exit(1)

    # --- Determine Suites and Regeneration Mode ---
    suites_to_regenerate = []
    regenerate_all = False
    needs_confirmation = False

    if args.suites:
        # User specified specific suites
        regenerate_all = False
        needs_confirmation = False  # Never confirm for specific suites
        invalid_suites = []
        requested_suites_set = set(args.suites)  # Handle duplicates
        for suite in requested_suites_set:
            if suite in available_suites:
                suites_to_regenerate.append(suite)
            else:
                invalid_suites.append(suite)

        if invalid_suites:
            logging.error(f"Unknown suite(s): {', '.join(invalid_suites)}")
            logging.info(available_suites_str)
            sys.exit(1)

        if args.force:
            # Force flag is only relevant when regenerating all, provide feedback if misused.
            logging.warning(
                "--force/-f flag is ignored when specific suites are provided."
            )

        logging.info(
            f"Regenerating data for specific suites: {', '.join(sorted(suites_to_regenerate))}"
        )

    else:
        # User did not specify suites - regenerate all
        regenerate_all = True
        suites_to_regenerate = sorted(
            list(available_suites.keys())
        )  # Sort for consistent order

        if not args.force:
            # Only need confirmation if regenerating all AND --force is not used
            needs_confirmation = True
        else:
            # User specified --force when regenerating all
            logging.warning(
                f"Proceeding with --force/-f: Removing entire data directory '{DATA_DIR.resolve()}' without confirmation."
            )

    # --- Perform Confirmation if Needed ---
    if needs_confirmation:
        try:
            print("-" * 60)
            print("WARNING: This script will DELETE the entire directory:")
            print(f"  '{DATA_DIR.resolve()}'")
            print("and regenerate data for ALL available suites:")
            print(f"  {', '.join(suites_to_regenerate)}")
            print("-" * 60)
            confirm = input("Proceed? (y/N): ")
            if confirm.lower() != "y":
                logging.info("Operation cancelled by user.")
                sys.exit(0)
        except EOFError:  # Handle non-interactive environments
            logging.error(
                "Confirmation required, but running in non-interactive mode. Use --force/-f to proceed."
            )
            sys.exit(1)

    # --- Perform Deletion ---
    if regenerate_all:
        if DATA_DIR.exists():
            logging.info(f"Removing entire data directory: {DATA_DIR}")
            try:
                shutil.rmtree(DATA_DIR)
            except OSError as e:
                logging.error(f"Error removing directory {DATA_DIR}: {e}")
                sys.exit(1)
        # Recreate base data directory
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logging.error(f"Error creating directory {DATA_DIR}: {e}")
            sys.exit(1)
    else:
        # Delete only specified suite directories
        for suite in suites_to_regenerate:
            suite_data_dir = DATA_DIR / suite
            if suite_data_dir.exists():
                logging.info(f"Removing suite data directory: {suite_data_dir}")
                try:
                    shutil.rmtree(suite_data_dir)
                except OSError as e:
                    logging.error(f"Error removing directory {suite_data_dir}: {e}")
                    sys.exit(1)
            # Suite directory will be recreated by the script's save_data helper

    # --- Perform Regeneration ---
    logging.info("Starting data generation...")
    success_count = 0
    for suite in suites_to_regenerate:
        script_path = available_suites[suite]
        run_script(script_path)  # run_script handles errors and exits on failure
        success_count += 1

    logging.info("-" * 60)
    logging.info(f"Successfully regenerated data for {success_count} suite(s).")
    logging.info("-" * 60)


if __name__ == "__main__":
    main()
