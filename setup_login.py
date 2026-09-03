from playwright.sync_api import sync_playwright

from config import BROWSER_PROFILE_DIR


# Use any actual NPTEL course link.
# Replace this with one of your course links if needed.
NPTEL_URL = (
    "https://onlinecourses.nptel.ac.in/"
)


def main():

    print("Starting persistent browser...")

    with sync_playwright() as p:

        # This is the important part.
        # Everything you do in this browser is saved in:
        # browser_profile/
        context = p.chromium.launch_persistent_context(
            user_data_dir=str(BROWSER_PROFILE_DIR),
            headless=False,
            viewport={
                "width": 1400,
                "height": 900
            }
        )

        page = context.pages[0]

        print("Opening NPTEL...")

        page.goto(
            NPTEL_URL,
            wait_until="domcontentloaded"
        )

        print("\n" + "=" * 60)
        print("MANUAL LOGIN REQUIRED")
        print("=" * 60)

        print("""
1. Log into NPTEL
2. Click Google login
3. Select your college Google account
4. Complete login
5. Make sure you can access your course
6. Return here
7. Press ENTER to save the session
        """)

        input("Press ENTER after successful login...")

        print("Saving browser session...")

        context.close()

        print("Done.")
        print(
            f"Persistent browser profile saved at:\n"
            f"{BROWSER_PROFILE_DIR}"
        )


if __name__ == "__main__":
    main()
