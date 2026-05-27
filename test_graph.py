from services.sharepoint_graph import SharePointGraph
from config import TENANT_ID, CLIENT_ID, CLIENT_SECRET

client = SharePointGraph(TENANT_ID, CLIENT_ID, CLIENT_SECRET)

print(client.token[:50])