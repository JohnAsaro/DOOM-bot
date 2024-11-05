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
    files_folder_path = os.path.join(os.getcwd(), "ViZDoom")
    
    #Check if the file exists in the 'ViZDoom' folder
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
