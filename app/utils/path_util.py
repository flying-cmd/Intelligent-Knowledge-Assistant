# app/utils/path_utils.py
from pathlib import Path
from dotenv import load_dotenv
import os
from pathlib import Path

def get_path_dir(ps:int = 0)->Path:
    """
    `pathlib.Path` provides a `parents` sequence, which makes it easy to fetch
    an ancestor directory by index instead of chaining `.parent` repeatedly.
    Rule of thumb: `parents[N]` returns the `(N + 1)`th parent.
    :param ps:
    :return:
    """
    dir_path = Path(__file__).parents[ps]
    return dir_path


def get_project_root(identifier: str = ".env") -> Path:
    # Step 1: Prefer the environment variable when available.
    env_root = os.getenv("PROJECT_ROOT")
    if env_root and Path(env_root).absolute().exists():
        return Path(env_root).absolute()

    # Step 2: Load the root `.env` file if we can find it.
    current_dir = Path(__file__).absolute().parent
    while current_dir != current_dir.parent:
        if (current_dir / identifier).exists():
            load_dotenv(dotenv_path=current_dir / identifier)
            break
        current_dir = current_dir.parent

    # Step 3: Recursively search for the identifier as a fallback.
    current_dir = Path(__file__).absolute().parent
    while current_dir != current_dir.parent:
        if (current_dir / identifier).exists():
            return current_dir
        current_dir = current_dir.parent

    raise FileNotFoundError(
        f"Could not find the project root identifier '{identifier}', and PROJECT_ROOT is not configured."
    )


PROJECT_ROOT = get_project_root(".env")
