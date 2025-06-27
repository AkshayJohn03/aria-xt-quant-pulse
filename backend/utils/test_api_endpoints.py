import requests
import json

BASE_URL = "http://localhost:8000/api/v1"

endpoints = [
    ("Market Data", f"{BASE_URL}/market-data?symbol=NIFTY50"),
    ("Portfolio", f"{BASE_URL}/portfolio"),
    ("Connection Status", f"{BASE_URL}/connection-status"),
]

def pretty_print(title, data):
    print(f"\n=== {title} ===")
    try:
        print(json.dumps(data, indent=2))
    except Exception:
        print(data)

def main():
    for name, url in endpoints:
        try:
            resp = requests.get(url, timeout=10)
            print(f"\n{name} [{resp.status_code}]: {url}")
            if resp.status_code == 200:
                data = resp.json()
                pretty_print(name, data)
                # Check for mock or error
                if isinstance(data, dict):
                    if data.get('source') == 'mock':
                        print("[WARNING] Mock data detected!")
                    if data.get('status') == 'error' or data.get('status') == 'offline':
                        print("[ERROR] Endpoint returned error/offline status!")
            else:
                print(f"[ERROR] Status code: {resp.status_code}")
                print(resp.text)
        except Exception as e:
            print(f"[EXCEPTION] {name}: {e}")

if __name__ == "__main__":
    main() 