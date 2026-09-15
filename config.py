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

# Optional proxy for YouTube transcript fetching, e.g. a local
# Tor SOCKS5 proxy: socks5h://127.0.0.1:9050
# Leave unset to fetch directly (default behavior).
YT_PROXY = os.getenv("YT_PROXY") or None

# Whether the automated NPTEL scraping browsers (main.py's
# video/transcript step, ingestion/assignment.py's assignment
# scrape) run headless. Single place to flip when you need to
# watch what the scraper actually sees — set HEADLESS=false in
# .env. Does not affect auth/setup_login.py's manual login window,
# which is always visible on purpose.
HEADLESS = os.getenv("HEADLESS", "true").strip().lower() not in (
    "false", "0", "no"
)

AUTH_DIR.mkdir(exist_ok=True)
BROWSER_PROFILE_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
