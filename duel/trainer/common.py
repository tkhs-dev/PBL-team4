import os
import subprocess

def file_exists_and_not_empty(file_path: str) -> bool:
    return os.path.exists(file_path) and os.stat(file_path).st_size != 0

def get_commit_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()

def is_dirty():
    try:
        # Run the git status command to check for changes
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        # If there is output, the repository has uncommitted changes
        return bool(result.stdout.strip())
    except subprocess.CalledProcessError:
        # If an error occurs, assume the repository is not accessible or not dirty
        return False

def get_version_str(version:str) -> str:
    result = f"v{version}-{get_commit_hash()}"
    if is_dirty():
        result += "-dirty"
    return result