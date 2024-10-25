#Class to create a ViZDoom environment for OpenAI Gym

from vizdoom import * #Import all of vizdoom
import numpy as np #Numpy for identity matrix
import time #To make the program sleep (wait), so we can actually see what's happening
from gymnasium import Env #Import OpenAI Gym's Env class
from gymnasium.spaces import Discrete, Box #Import OpenAI Gym's Discrete and Box spaces
import cv2 #OpenCV for image processing, used for modifying the DOOM environment to make it run faster 
from stable_baselines3.common.callbacks import BaseCallback #Import the BaseCallback class from stable_baselines3 to learn from the environment
import os #To create directories for saving models

class Deadly_Corridor_VZG(Env): #VizDoom + OpenAI Gym environment for deadly corridor
    
    #We now have scenarios that are complicated enough where we can't just use a generalized enviornment
    #so this one is made for deadly corridor, however, this can probably be reused for something else,
    #so the naming convention will be "First map this was used on" + "_VZG" (VizDoomGym), 
    #but each config/map this env is used for will be listed below

    #Maps/Config: deadly_corridor

    def __init__(self, config_path, render=False): #Constructor
        
        super(Deadly_Corridor_VZG, self).__init__() #Inherit from Env class

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
        self.action_space = Discrete(7) #Action space, 3 actions

        #Game variables
        self.hitcount = 0
        self.ammo = 52 #We only have the pistol in this scenario and we start with 52 ammo
        self.damage_taken = 0

    def step(self, action, limit = 1000): #Take a step in the environment 
        #Args:
            #action (int): The action to take
            #limit (int): Unimplemented "limit" for the episode, most likely will be a time limit
        #Returns:
            #observation (np.array): The screen buffer of the environment
            #reward (float): The reward for the action taken
            #terminated (bool) Whether the episode is finished or not (by reaching the goal)
            #truncated (bool): Whether the episode has reached some terminal state without reaching the goal (ie: running out of time)
            #info (dict): Additional information about the environment

        #Specify actions and take a step
        actions = np.identity(7) #Create an identity matrix with 3 rows (3 actions), MOVE_LEFT, MOVE_RIGHT, ATTACK, these are the actions we can take in the environment
        movement_reward = self.game.make_action(actions[action], 4) #Reward for taking a random action, second parameter is frame skip (skip 4 frames before taking the next action), the reason we do this is because it saves us time while being easy to see what is happening 
        reward = 0 #Initialize reward to 0
        truncated = False #Not implemented yet, so set to False. The idea is that if step passes some sort of limit, like a time limit, then the episode is truncated.
        info = {} #Initialize info to an empty dictionary

        if self.game.get_state(): #If the game is not finished
            observation = self.game.get_state().screen_buffer #Get the screen buffer
            observation = self.greyscale(observation) #Convert the image to greyscale

            #Get game variables
            game_variables = self.game.get_state().game_variables
            health, damage_taken, hitcount, ammo = game_variables

            #Calculate reward deltas
            ammo_delta = ammo - self.ammo #Current ammo - old ammo = ammo used
            ammo = self.ammo 
            damage_taken_delta = -damage_taken + self.damage_taken  #Current negative damage taken + old damage taken = damage taken
            self.damage_taken = damage_taken 
            hitcount_delta = hitcount - self.hitcount #Current hitcount - old hitcount = hits taken
            self.hitcount = hitcount
            
            reward = movement_reward + damage_taken_delta*10 + ammo_delta*5 + hitcount_delta*200 #Calculate the reward
            info = {"ammo": ammo}
        else:
            observation = np.zeros(self.observation_space.shape) #Return a blank screen

        terminated = self.game.is_episode_finished() #Check if the episode is finished

        return observation, reward, terminated, truncated, info

    def render(self, render_in_greyscale=False): #Render the environment for a frame
        #Args:
            #render_in_greyscale (bool): Whether to render the environment in greyscale or not
        
        if self.game.get_state() and render_in_greyscale:  #Only render if there's a valid game state
            observation = self.game.get_state().screen_buffer
            greyscale_obs = self.greyscale(observation)  #Convert to greyscale
            #Render using OpenCV to visualize
            cv2.imshow("VizDoom Environment", greyscale_obs.squeeze())  #Remove extra dimension and display
            cv2.waitKey(1)  #Wait 1ms between frames to allow for rendering
        elif self.game.get_state():  #Only render if there's a valid game state
            observation = self.game.get_state().screen_buffer
            #Render using OpenCV to visualize
            cv2.imshow("VizDoom Environment", observation.squeeze())  #Remove extra dimension and display
            cv2.waitKey(1)  #Wait 1ms between frames to allow for rendering
        else:
            print("No game state to render.")

            
    def reset(self, seed=None): #Reset the environment when we start a new game
        #Args:
            #seed (int): The seed for the random number generator
        #Returns:
            #(observation, info) (tuple)
                #observation (np.array): The screen buffer of the environment
                #info (dict): Additional information about the environment
            
        super().reset(seed=seed) #Implement seeding
        
        self.game.new_episode() #Start a new episode
        state = self.game.get_state().screen_buffer #Get the screen buffer
        observation = self.greyscale(state) #Convert the image to greyscale
        
        #Gather any additional environment-specific info (like ammo, etc.)
        if self.game.get_state():
            ammo = self.game.get_state().game_variables[0]  # Get the ammo count                
            info = {"ammo": ammo}
        else:
            info = {} #No gamestate means no info can be gathered
        
        return (observation, info) #Tuple of observation and info

    def greyscale(self, observation=None): #Convert the enivornment to greyscale and resize it
        #Args:
            #observation (np.array): The image of the environment (the current game frame)
        #Returns:
            #grey_return (np.array): The resized greyscale image of the environment
        
        if observation is None and self.game.get_state(): #If no observation is passed
            observation = self.game.get_state().screen_buffer #Get the screen buffer 

        grey = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY) #Convert the image to greyscale
        resize = cv2.resize(grey, (160, 100), interpolation=cv2.INTER_CUBIC) #Resize the image to 160x100
        state = np.reshape(resize, (100, 160, 1)) #Reshape the image to 100x160x1
        
        return state
    
    def get_state(self): 
        #Returns:
            #state (np.array): The current state of the environment
        return self.game.get_state()

    def close(self): #Close the environment
        self.game.close()

class VizDoomGym_Simple(Env): #Old Simplified VizDoom + OpenAI Gym environment, used for defend_the_center and basic configs
    def __init__(self, config_path, render=False): #Constructor
        
        #Configs this is used for: basic.cfg, defend_the_center.cfg

        super(VizDoomGym_Simple, self).__init__() #Inherit from Env class

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

        #Game variables
        self.ammo = self.game.get_state().game_variables[0]  #Get the ammo count, initialize to the current ammo
        self.health = 100 #Initialize health to 100 (assuming we start at full health)


    def step(self, action, limit = 1000): #Take a step in the environment 
        #Args:
            #action (int): The action to take
            #limit (int): Unimplemented "limit" for the episode, most likely will be a time limit
        #Returns:
            #observation (np.array): The screen buffer of the environment
            #reward (float): The reward for the action taken
            #terminated (bool) Whether the episode is finished or not (by reaching the goal)
            #truncated (bool): Whether the episode has reached some terminal state without reaching the goal (ie: running out of time)
            #info (dict): Additional information about the environment

        #Specify actions and take a step
        actions = np.identity(3) #Create an identity matrix with 3 rows (3 actions), MOVE_LEFT, MOVE_RIGHT, ATTACK, these are the actions we can take in the environment
        movement_reward = self.game.make_action(actions[action], 4) #Reward for taking a random action, second parameter is frame skip (skip 4 frames before taking the next action), the reason we do this is because it saves us time while being easy to see what is happening 
        reward = movement_reward #Initialize reward to movement reward
        truncated = False #Not implemented yet, so set to False. The idea is that if step passes some sort of limit, like a time limit, then the episode is truncated.
        info = {} #Initialize info to an empty dictionary
        Basic = False

        if self.game.get_state(): #If the game is not finished
            observation = self.game.get_state().screen_buffer #Get the screen buffer
            observation = self.greyscale(observation) #Convert the image to greyscale

            #Get game variables
            game_variables = self.game.get_state().game_variables
            if len(game_variables) == 2:
                ammo, health = game_variables
            else:
                ammo = game_variables[0]
                health = 100 #Assume health is 100 if it's not provided (effectively ignoring it)
                Basic = True
            
            #Calculate reward deltas
            if(Basic == False): #If its the basic config we just ignore all deltas entirely, I know this is janky but its just a testing enviorment so whatever
                ammo_delta = ammo - self.ammo #Current ammo - old ammo = ammo used
                ammo = self.ammo 
                health_delta = health - self.health  #Current health - old health = damage taken
                health = self.health
            
                reward = movement_reward + ammo_delta*0.01 + health_delta*0.02 #Calculate the reward
            
            info = {"ammo": ammo}
        else:
            observation = np.zeros(self.observation_space.shape) #Return a blank screen

        terminated = self.game.is_episode_finished() #Check if the episode is finished

        return observation, reward, terminated, truncated, info

    def render(self, render_in_greyscale=False): #Render the environment for a frame
        #Args:
            #render_in_greyscale (bool): Whether to render the environment in greyscale or not
        
        if self.game.get_state() and render_in_greyscale:  #Only render if there's a valid game state
            observation = self.game.get_state().screen_buffer
            greyscale_obs = self.greyscale(observation)  #Convert to greyscale
            #Render using OpenCV to visualize
            cv2.imshow("VizDoom Environment", greyscale_obs.squeeze())  #Remove extra dimension and display
            cv2.waitKey(1)  #Wait 1ms between frames to allow for rendering
        elif self.game.get_state():  #Only render if there's a valid game state
            observation = self.game.get_state().screen_buffer
            #Render using OpenCV to visualize
            cv2.imshow("VizDoom Environment", observation.squeeze())  #Remove extra dimension and display
            cv2.waitKey(1)  #Wait 1ms between frames to allow for rendering
        else:
            print("No game state to render.")

            
    def reset(self, seed=None): #Reset the environment when we start a new game
        #Args:
            #seed (int): The seed for the random number generator
        #Returns:
            #(observation, info) (tuple)
                #observation (np.array): The screen buffer of the environment
                #info (dict): Additional information about the environment
            
        super().reset(seed=seed) #Implement seeding
        
        self.game.new_episode() #Start a new episode
        state = self.game.get_state().screen_buffer #Get the screen buffer
        observation = self.greyscale(state) #Convert the image to greyscale
        
        #Gather any additional environment-specific info (like ammo, etc.)
        if self.game.get_state():
            ammo = self.game.get_state().game_variables[0]  #Get the ammo count
            info = {"ammo": ammo}
        else:
            info = {} #No gamestate means no info can be gathered
        
        return (observation, info) #Tuple of observation and info

    def greyscale(self, observation=None): #Convert the enivornment to greyscale and resize it
        #Args:
            #observation (np.array): The image of the environment (the current game frame)
        #Returns:
            #grey_return (np.array): The resized greyscale image of the environment
        
        if observation is None and self.game.get_state(): #If no observation is passed
            observation = self.game.get_state().screen_buffer #Get the screen buffer 

        grey = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY) #Convert the image to greyscale
        resize = cv2.resize(grey, (160, 100), interpolation=cv2.INTER_CUBIC) #Resize the image to 160x100
        state = np.reshape(resize, (100, 160, 1)) #Reshape the image to 100x160x1
        
        return state
    
    def get_state(self): 
        #Returns:
            #state (np.array): The current state of the environment
        return self.game.get_state()

    def close(self): #Close the environment
        self.game.close()

#Example usage
#from pathfinder import doomfinder #To find stuff in the ViZDoom folder
#game_environment = VizDoomGym_Simple(doomfinder("basic.cfg"), True) #Create a ViZDoom environment

class TrainAndLogCallback(BaseCallback):
    
    def __init__(self, check_freq, save_path, verbose = 1):

        #Args:
            #check_freq (int): Frequency to check the model
            #save_path (str): Path to save the model
            #verbose (int): Verbosity level, 1 by default

        super(TrainAndLogCallback, self).__init__(verbose)
        self.check_freq = check_freq #Frequency to check the model
        self.save_path = save_path #Path to save the model

    def _init_callback(self): #Initialize the callback
        if self.save_path is not None: 
            os.makedirs(self.save_path, exist_ok=True) #Create the save path if it doesn't exist
        
    def _on_step(self): #Check the model after each step
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls)) #Might change this to be a pathfinder function later if I use it more than once
            self.model.save(model_path)
        
        return True