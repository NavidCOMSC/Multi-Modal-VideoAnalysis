from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from os import getenv

ENVIRONMENT = getenv("environment", "dev")
URI = f"https://ida-{ENVIRONMENT}-kv.vault.azure.net/"
CREDENTIAL = DefaultAzureCredential()

secret_client = SecretClient(vault_url=URI, credential=CREDENTIAL)
