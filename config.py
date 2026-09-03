from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

BROWSER_PROFILE_DIR = BASE_DIR / "browser_profile"
DATA_DIR = BASE_DIR / "data"
SCREENSHOTS_DIR = BASE_DIR / "screenshots"

BROWSER_PROFILE_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
SCREENSHOTS_DIR.mkdir(exist_ok=True)

