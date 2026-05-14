import os
import requests
from pathlib import Path

URL = "https://github.com/ThetaCursed/Anima-TrainFlow/blob/main/app.py"
LOCAL_FILE = Path("app.py")
ETAG_FILE = Path(".app_etag")

def update():
    headers = {}
    
    if ETAG_FILE.exists():
        etag = ETAG_FILE.read_text().strip()
        headers["If-None-Match"] = etag
    
    try:
        response = requests.get(URL, headers=headers, timeout=15)

        if response.status_code == 304:
            print("Current version is up to date.")
            
        elif response.status_code == 200:
            print("Updating app.py...")
            with open(LOCAL_FILE, "wb") as f:
                f.write(response.content)
            

            new_etag = response.headers.get("ETag")
            if new_etag:
                ETAG_FILE.write_text(new_etag)
            print("✨ Successfully updated!")
            
        else:
            print(f"⚠️ Server returned status: {response.status_code}")

    except Exception as e:
        print(f"❌ Connection error: {e}")

if __name__ == "__main__":
    update()