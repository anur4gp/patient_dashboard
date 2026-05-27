import msal
import requests
import pandas as pd
import io


class SharePointGraph:
    def __init__(self, tenant_id, client_id, client_secret):
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret

        self.token = self._get_token()

    def _get_token(self):
        authority = f"https://login.microsoftonline.com/{self.tenant_id}"

        app = msal.ConfidentialClientApplication(
            self.client_id,
            authority=authority,
            client_credential=self.client_secret
        )

        result = app.acquire_token_for_client(
            scopes=["https://graph.microsoft.com/.default"]
        )

        if "access_token" not in result:
            raise Exception(result)

        return result["access_token"]

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.token}"
        }
    
    def download_excel(self, drive_item_id):
        url = f"https://graph.microsoft.com/v1.0/me/drive/items/{drive_item_id}/content"

        r = requests.get(url, headers=self._headers())

        if r.status_code != 200:
            raise Exception(r.text)

        return pd.read_excel(io.BytesIO(r.content))
    
    def list_drives(self):
        url = "https://graph.microsoft.com/v1.0/sites/root/drives"

        r = requests.get(url, headers=self._headers())

        if r.status_code != 200:
            raise Exception(r.text)

        return r.json()