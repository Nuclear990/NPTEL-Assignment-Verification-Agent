from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
AUTH_DIR = BASE_DIR / "auth"

BROWSER_PROFILE_DIR = AUTH_DIR / "browser_profile"
DATA_DIR = BASE_DIR / "data"
SCREENSHOTS_DIR = BASE_DIR / "screenshots"

CREDENTIALS_FILE = AUTH_DIR / "credentials.json"
TOKEN_FILE = AUTH_DIR / "token.json"

AUTH_DIR.mkdir(exist_ok=True)
BROWSER_PROFILE_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
SCREENSHOTS_DIR.mkdir(exist_ok=True)

