# test_api.py

import requests

response = requests.post(
    "http://127.0.0.1:8000/ask",
    json={
        "query":"What are symptoms of panic attack vs. anxiety attack?"
    }
)

print(response.json())