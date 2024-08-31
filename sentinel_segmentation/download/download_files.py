"""
Author: Yibing Chen
GitHub username: edsml-yc4523
"""

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import os
import io

# If modifying these SCOPES, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/drive']


def authenticate_gdrive(credentials_path, token_path):
    """
    Authenticate and create a Google Drive service.

    Args:
        credentials_path (str):
                The path to the Google API credentials JSON file.
        token_path (str):
                The path to the token JSON file for storing user credentials.

    Returns:
        googleapiclient.discovery.Resource: A resource object with methods for
        interacting with the service.

    Raises:
        Exception: If the authentication process fails for any reason.
    """
    creds = None
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                credentials_path, SCOPES
            )
            creds = flow.run_local_server(port=0)
        with open(token_path, 'w') as token:
            token.write(creds.to_json())
    return build('drive', 'v3', credentials=creds)


def download_file(service, file_id, file_name):
    """
    Download a single file from Google Drive.

    Given a file ID, this function downloads the file from Google Drive and
    saves it to the specified local path.

    Args:
        service (googleapiclient.discovery.Resource): The authenticated
            Google Drive service instance.
        file_id (str): The ID of the file to download.
        file_name (str): The local path where the file should be saved.

    Returns:
        None

    Raises:
        googleapiclient.errors.HttpError: If there is an issue with the
        HTTP request to the Google Drive API.
    """
    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(file_name, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        print(f"Download {int(status.progress() * 100)}%.")
    fh.close()


def download_folder(service, folder_id, destination):
    """
    Download all files from a Google Drive folder.

    This function downloads all files from a specified Google Drive folder,
    preserving the folder structure.

    Args:
        service (googleapiclient.discovery.Resource): The authenticated
            Google Drive service instance.
        folder_id (str): The ID of the folder to download.
        destination (str): The local directory where the folder contents
            should be saved.

    Returns:
        None

    Raises:
        googleapiclient.errors.HttpError: If there is an issue with the
        HTTP request to the Google Drive API.
        OSError: If there is an issue creating the local directory structure.
    """
    query = f"'{folder_id}' in parents"
    results = service.files().list(
        q=query, fields="files(id, name, mimeType)"
    ).execute()
    items = results.get('files', [])

    if not os.path.exists(destination):
        os.makedirs(destination)

    for item in items:
        file_id = item['id']
        file_name = item['name']
        file_path = os.path.join(destination, file_name)
        if item['mimeType'] == 'application/vnd.google-apps.folder':
            download_folder(service, file_id, file_path)
        else:
            download_file(service, file_id, file_path)
