from services.sharepoint_graph import SharePointGraph
from config import TENANT_ID, CLIENT_ID, CLIENT_SECRET
import json

client = SharePointGraph(TENANT_ID, CLIENT_ID, CLIENT_SECRET)

drives = client.list_drives()

print(json.dumps(drives, indent=2))