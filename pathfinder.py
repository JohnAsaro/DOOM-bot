import os

def doomfinder(filename):
    """
    Finds the given file in the 'ViZDoom' folder or prompts the user to provide a path.

    Args:
        filename (str): The name of the file to find.

    Returns:
        str: The absolute path to the file.

    Raises:
        FileNotFoundError: If the file cannot be found.
    """
    #Define the path to the 'Files' folder
    files_folder_path = os.path.join(os.getcwd(), "ViZDoom")

    #Check if the file exists in the 'Files' folder
    file_path = os.path.join(files_folder_path, filename)
    if os.path.isfile(file_path):
        return os.path.abspath(file_path)
    else:
        print(f"File '{filename}' not found in the 'ViZDoom' folder.")
        #Prompt the user to provide a path
        user_file_path = input("Please enter the full path to the file: ")
        if os.path.isfile(user_file_path):
            return os.path.abspath(user_file_path)
        else:
            raise FileNotFoundError(f"File '{user_file_path}' not found.")