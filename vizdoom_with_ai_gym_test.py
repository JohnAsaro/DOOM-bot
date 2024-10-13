#Class to create a ViZDoom environment for OpenAI Gym

from vizdoom import * #Import all of vizdoom
import numpy as np #Numpy for identity matrix
import time #To make the program sleep (wait), so we can actually see what's happening
from gym import Env #Import OpenAI Gym's Env class
from gym.spaces import Discrete, Box #Import OpenAI Gym's Discrete and Box spaces
import cv2 #OpenCV for image processing, used for modifying the DOOM environment to make it run faster 

class VizDoomGym(Env): 
    def __init__(self, config_path, render=False): #Constructor
        
        super().__init__() #Inherit from Env class

        #Args: 
            #config_path (str): The path to the configuration file
            #render (bool): Whether to render the environment or not, false by default

        #Setup game
        self.game = vizdoom.DoomGame() #Create a DoomGame object
        self.game.load_config(config_path) #Load the configuration file from file path, ex: doomfinder("basic.cfg")

        #Set window visibility
        if render == False:
            self.game.set_window_visible(False)
        else:
            self.game.set_window_visible(True)
    
        self.game.init() #Start the game

        #Setup action and observation space
        self.observation_space = Box(low=0, high=255, shape=(100, 160, 1), dtype=np.uint8) #Observation space, 100x160x1 image
        self.action_space = Discrete(3) #Action space, 3 actions

    def step(self, action): #Take a step in the environment 
        #Args:
            #action (int): The action to take
        #Returns:
            #state (np.array): The screen buffer of the environment
            #reward (float): The reward for the action taken
            #done (bool): Whether the episode is finished or not
            #info (dict): Additional information about the environment

        #Specify actions and take a step
        actions = np.identity(3) #Create an identity matrix with 3 rows (3 actions), MOVE_LEFT, MOVE_RIGHT, ATTACK, these are the actions we can take in the environment
        reward = self.game.make_action(actions[action], 4) #Take a random action, second parameter is frame skip (skip 4 frames before taking the next action), the reason we do this is because it saves us time while being easy to see what is happening 

        if self.game.get_state(): #If the game is not finished
            state = self.game.get_state().screen_buffer #Get the screen buffer
            state = self.greyscale(state) #Convert the image to greyscale
            ammo = self.game.get_state().game_variables[0] #Get the game variables (ammo in this case)
            info = ammo #Return the ammo count
        else:
            state = np.zeros(self.observation_space.shape) #Return a blank screen
            info = 0 #Return 0 if the game is finished

        info = {"info": info} #Return the info as a dictionary
        done = self.game.is_episode_finished() #Check if the episode is finished

        return state, reward, done, info
    
    def render(): #Render the environment
        pass
    
    def reset(self): #Reset the environment when we start a new game
        #Returns:
            #state (np.array): The screen buffer of the environment
        
        self.game.new_episode() #Start a new episode
        state = self.game.get_state().screen_buffer #Get the screen buffer
        return self.greyscale(state) #Return greyscale version of the screen
    
    def greyscale(self, observation): #Convert the enivornment to greyscale and resize it
        #Args:
            #observation (np.array): The image of the environment (the current game frame)
        #Returns:
            #grey_return (np.array): The resized greyscale image of the environment

        grey = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY) #Convert the image to greyscale
        resize = cv2.resize(grey, (160, 100), interpolation=cv2.INTER_CUBIC) #Resize the image to 160x100
        state = np.reshape(resize, (100, 160, 1)) #Reshape the image to 100x160x1
        
        return state
    
    def close(self): #Close the environment
        self.game.close()

#Example usage
#from pathfinder import doomfinder #To find stuff in the ViZDoom folder
#game_environment = VizDoomGym(doomfinder("basic.cfg"), True) #Create a ViZDoom environment
