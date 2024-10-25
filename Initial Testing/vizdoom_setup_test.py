from vizdoom import * #Import all of vizdoom
import numpy as np #Numpy for identity matrix
from pathfinder import doomfinder #To find stuff in the ViZDoom folder
import random #To generate random numbers
import time #To make the program sleep (wait), so we can actually see what's happening

#Start the game
game = vizdoom.DoomGame() #Create a DoomGame object
game.load_config(doomfinder("basic.cfg")) #Load the configuration file
game.init() #Start the game

actions = np.identity(3, dtype=np.uint) #Create an identity matrix with 3 rows (3 actions), MOVE_LEFT, MOVE_RIGHT, ATTACK, these are the actions we can take in the enviornment
random.choice(actions) #Choose a random action

episodes = 10 #Number of episodes (game sessions) to run
for episode in range(episodes):
    game.new_episode() #Start a new episode
    while not game.is_episode_finished(): #While the episode is not finished
        state = game.get_state() #Get the current state
        img = state.screen_buffer #Get the screen buffer
        info = state.game_variables #Get the game variables (ammo in this case)
        reward = game.make_action(random.choice(actions), 4) #Take a random action, second parameter is frame skip (skip 4 frames before taking the next action), the reason we do this is because it saves us time while being easy to see what is happening 
        print("\treward:", reward) #Print the reward
        time.sleep(0.02) #Wait for 0.02 seconds
    print("Result:", game.get_total_reward()) #Print the total reward
    time.sleep(2) #Wait for 2 seconds
