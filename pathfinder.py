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
    #Define the path to the 'ViZDoom' folder
    files_folder_path = os.path.join(os.getcwd(), "Maps and Configs")
    #Check if the file exists in the 'ViZDoom' folder
    file_path = os.path.join(files_folder_path, filename)
    if os.path.isfile(file_path):
        return os.path.abspath(file_path)
    else:
        print(f"File '{filename}' not found in the 'Maps and Configs' folder.")
        #Prompt the user to provide a path
        user_file_path = input("Please enter the full path to the file: ")
        if os.path.isfile(user_file_path):
            return os.path.abspath(user_file_path)
        else:
            raise FileNotFoundError(f"File '{user_file_path}' not found.")

def logfinder(filename):
    """
    Finds the given file in the 'Logs' folder or prompts the user to provide a path.

    Args:
        filename (str): The name of the file to find.

    Returns:
        str: The absolute path to the file.
    """
    #Define the path to the 'Logs' folder
    files_folder_path = os.path.join(os.getcwd(), "Logs")

    #Check if the file exists in the 'ViZDoom' folder
    file_path = os.path.join(files_folder_path, filename)
    if os.path.isfile(file_path):
        return os.path.abspath(file_path)
    else:
        print(f"File '{filename}' not found in the 'Logs' folder.")
        #Prompt the user to provide a path
        user_file_path = input("Please enter the full path to the file: ")
        if os.path.isfile(user_file_path):
            return os.path.abspath(user_file_path)
        else:
            raise FileNotFoundError(f"File '{user_file_path}' not found.")
        
def create_new_best_generation_directory(folder_name):
    """
    Creates a new directory for saving model checkpoints.

    Args:
        folder_name (str): The name of the new directory.
    Returns:
        str: The path to the new directory.
    """
    base_dir = './Training/Best_Models'  #Define the base directory for saving models
    #List all existing directories that match the "best_models_x" pattern
    existing_dirs = [d for d in os.listdir(base_dir) if d.startswith(folder_name)]

    #Extract the numbers from the folder names and find the maximum number
    folder_nums = [int(d.split('_')[-1]) for d in existing_dirs if d.split('_')[-1].isdigit()]
    
    if folder_nums:
        new_folder_num = max(folder_nums) + 1  #Increment the folder number
    else:
        new_folder_num = 1  #If no directories exist, start with 1

    #Create the new folder name
    new_folder_name = f"{folder_name}_{new_folder_num}"
    new_folder_path = os.path.join(base_dir, new_folder_name)

    #Create the directory
    os.makedirs(new_folder_path, exist_ok=True)

    return new_folder_path  #Return the path for saving models

def create_new_checkpoint_directory(folder_name):
    """
    Creates a new directory for saving model checkpoints.

    Args:
        folder_name (str): The name of the new directory.
    Returns:
        str: The path to the new directory.
    """
    base_dir = './Training/checkpoints'
    #List all existing directories that match the "best_models_x" pattern
    existing_dirs = [d for d in os.listdir(base_dir) if d.startswith(folder_name)]

    #Extract the numbers from the folder names and find the maximum number
    folder_nums = [int(d.split('_')[-1]) for d in existing_dirs if d.split('_')[-1].isdigit()]
    
    if folder_nums:
        new_folder_num = max(folder_nums) + 1  #Increment the folder number
    else:
        new_folder_num = 1  #If no directories exist, start with 1

    #Create the new folder name
    new_folder_name = f"{folder_name}_{new_folder_num}"
    new_folder_path = os.path.join(base_dir, new_folder_name)

    #Create the directory
    os.makedirs(new_folder_path, exist_ok=True)

    return new_folder_path  #Return the path for saving models

def gamefinder(filename):
    """
    Finds the specified game file (e.g., 'DOOM.WAD' or 'DOOM2.WAD') in the 'Games' directory.
    
    Args:
        filename (str): The name of the game file to find (e.g., 'DOOM.WAD').

    Returns:
        str: The absolute path to the specified game file.

    Raises:
        FileNotFoundError: If the specified game file cannot be found in the 'Games' directory.
    """
    # Get the absolute path to the root directory
    root_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))  # Go up one directory level from current cwd
    
    # Define the path to the 'Games' folder relative to the root directory
    games_dir = os.path.join(root_dir, 'Games')
    
    # Construct the path to the specified game file
    game_file_path = os.path.join(games_dir, filename)
    
    # Check if the game file exists
    if os.path.isfile(game_file_path):
        return game_file_path
    else:
        raise FileNotFoundError(f"File '{filename}' not found in the 'Games' directory at {games_dir}")