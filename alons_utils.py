
def windows_to_wsl_path(windows_path):
    """
    Convert a Windows path to WSL path.

    Examples:
    - D:\\Projects\\data\\file.jpg -> /mnt/d/Projects/data/file.jpg
    - C:\\Users\\name\\Documents\\file.txt -> /mnt/c/Users/name/Documents/file.txt
    """
    # Check if the path is actually a Windows path
    if not windows_path or not isinstance(windows_path, str):
        return windows_path

    # Handle drive letter
    if len(windows_path) > 1 and windows_path[1] == ':':
        drive_letter = windows_path[0].lower()
        path_without_drive = windows_path[2:]

        # Replace backslashes with forward slashes
        path_with_correct_slashes = path_without_drive.replace('\\', '/')

        # Construct the WSL path
        wsl_path = f"/mnt/{drive_letter}{path_with_correct_slashes}"
        return wsl_path
    else:
        # If no drive letter, just replace the slashes
        return windows_path.replace('\\', '/')