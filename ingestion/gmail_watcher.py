import os
import base64

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from config import CREDENTIALS_FILE, TOKEN_FILE


# ============================================================
# CONFIGURATION
# ============================================================

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly"
]


# ============================================================
# AUTHENTICATION
# ============================================================

def get_gmail_service():

    creds = None

    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(
            TOKEN_FILE,
            SCOPES
        )

    if not creds or not creds.valid:

        refreshed = False

        if creds and creds.expired and creds.refresh_token:

            print("Refreshing Gmail credentials...")

            try:
                creds.refresh(Request())
                refreshed = True

            except RefreshError:
                print("Gmail token refresh failed (expired or revoked). Re-authenticating...")

        if not refreshed:

            print("Opening Google login...")

            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_FILE,
                SCOPES
            )

            creds = flow.run_local_server(port=0)

        with open(TOKEN_FILE, "w") as token:
            token.write(creds.to_json())

        print("Gmail authentication successful.")

    return build(
        "gmail",
        "v1",
        credentials=creds
    )


# ============================================================
# EMAIL BODY EXTRACTION
# ============================================================

def get_email_body(payload):

    if "parts" in payload:

        for part in payload["parts"]:

            body = get_email_body(part)

            if body:
                return body

    body_data = payload.get(
        "body",
        {}
    ).get(
        "data"
    )

    if body_data:

        decoded = base64.urlsafe_b64decode(
            body_data + "=" * (-len(body_data) % 4)
        )

        return decoded.decode(
            "utf-8",
            errors="ignore"
        )

    return ""


# ============================================================
# HEADER EXTRACTION
# ============================================================

def get_header(headers, name):

    for header in headers:

        if header["name"].lower() == name.lower():
            return header["value"]

    return None


# ============================================================
# GET NPTEL ASSIGNMENT ANNOUNCEMENT EMAILS
# ============================================================

def get_recent_nptel_emails(max_results=15):

    service = get_gmail_service()

    # Only fetch NPTEL emails announcing
    # that weekly content + assignment are live
    query = (
        'from:(onlinecourses@nptel.iitm.ac.in) '
        'subject:"Content & Assignment is live now" '
        'newer_than:30d'
    )

    results = service.users().messages().list(
        userId="me",
        q=query,
        maxResults=max_results
    ).execute()

    messages = results.get(
        "messages",
        []
    )

    if not messages:

        print("No NPTEL assignment announcement emails found.")
        return []

    emails = []

    for message_ref in messages:

        message_id = message_ref["id"]

        # Fetch complete email
        message = service.users().messages().get(
            userId="me",
            id=message_id,
            format="full"
        ).execute()

        headers = message["payload"].get(
            "headers",
            []
        )

        subject = get_header(
            headers,
            "Subject"
        )

        sender = get_header(
            headers,
            "From"
        )

        date = get_header(
            headers,
            "Date"
        )

        body = get_email_body(
            message["payload"]
        )

        emails.append({

            "id": message_id,

            "subject": subject,

            "sender": sender,

            "date": date,

            "body": body

        })

    return emails


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    emails = get_recent_nptel_emails(
        max_results=15
    )

    print("\n")
    print("=" * 80)
    print(f"FOUND {len(emails)} NPTEL ASSIGNMENT ANNOUNCEMENT EMAIL(S)")
    print("=" * 80)

    for index, email in enumerate(emails, start=1):

        print("\n")
        print("#" * 80)
        print(f"EMAIL {index}")
        print("#" * 80)

        print("\nID:")
        print(email["id"])

        print("\nFROM:")
        print(email["sender"])

        print("\nDATE:")
        print(email["date"])

        print("\nSUBJECT:")
        print(email["subject"])

        print("\n" + "-" * 80)
        print("BODY")
        print("-" * 80)

        print(email["body"])

        print("\n" + "#" * 80)
