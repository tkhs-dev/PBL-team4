import os
import tarfile

import requests


def setup():
    if os.name != "nt":
        print("This setup script is only for Windows. Please download the rules manually.")
        return
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if os.path.exists("rules/battlesnake.exe"):
        return
    print("Setting up Battlesnake environment")
    resp = requests.get("https://github.com/BattlesnakeOfficial/rules/releases/download/v1.2.3/battlesnake_1.2.3_Windows_x86_64.tar.gz")
    with open("rule.tar.gz", "wb") as f:
        f.write(resp.content)
    print(" - Downloaded Battlesnake rules")
    with tarfile.open('rule.tar.gz', 'r:gz') as tar:
        tar.extractall(path="rules")
    print(" - Extracted Battlesnake rules")
    os.remove("rule.tar.gz")
    print(" - Removed Battlesnake rules archive")
    print("Setup complete")

if __name__ == "__main__":
    setup()