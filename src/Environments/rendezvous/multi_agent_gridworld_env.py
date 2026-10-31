import random, math, os
import numpy as np
from enum import Enum

import sys
sys.path.append('../')
sys.path.append('../../')
from reward_machines.sparse_reward_machine import SparseRewardMachine

"""
Enum with the actions that the agent can execute
"""
class Actions(Enum):
    up    = 0 # move up
    right = 1 # move right
    down  = 2 # move down
    left  = 3 # move left
    change_color_up = 4
    change_color_down = 5
    change_shape_up = 6
    change_shape_down = 7 

class MultiAgentGridWorldEnv:

    def __init__(self, rm_file, num_agents, env_settings):
        """
        Initialize gridworld environment.

        Parameters
        ----------
        rm_file : string
            File path leading to the text file containing the reward machine
            encoding this environment's reward function.
        num_agents : int
            Number of agents in the environment.
        env_settings : dict
            Dictionary of environment settings
        """
        self.env_settings = env_settings
        self.num_agents = num_agents
        
        self._load_map()
        self.reward_machine = SparseRewardMachine(rm_file)

        self.u = self.reward_machine.get_initial_state()
        self.last_action = np.full(self.num_agents, -1, dtype=int) #Initialize last action with garbage values

        self.rendezvous_complete = False
        self.reached_goal_flag = []
        for i in range(num_agents):
            self.reached_goal_flag.append(False)

    def _load_map(self):
        """
        Initialize the environment.
        """
        self.Nr = self.env_settings['Nr']
        self.Nc = self.env_settings['Nc']
        num_shapes = 3
        num_colors = 3

        initial_states = self.env_settings['initial_states']

        # Define Initial states of all agents (only x, y coordinates)
        self.s_i = np.full(self.num_agents, -1, dtype=int)
        for i in range(self.num_agents):
            x, y = initial_states[i]['coords']
            shape_index = initial_states[i].get('shape', 0)  # Default to 0 if no shape
            color_index = initial_states[i].get('color', 0)  # Default to 0 if no color

            self.s_i[i] = (x * self.Nc + y) * (num_shapes * num_colors) + shape_index * num_colors + color_index

        # Populate the map with markers for objects and locations
        self.objects = {}

        # Add rendezvous location (no shape or color, just a marker)
        # rendezvous_loc = self.env_settings['rendezvous_loc']
        # self.objects[(rendezvous_loc[0], rendezvous_loc[1])] = "w"  # Only store the (x, y) position

        # # Add goal locations for agents (x, y only)
        # goal_locations = self.env_settings['goal_locations']
        # for i in range(self.num_agents):
        #     goal_loc = goal_locations[i]
        #     self.objects[(goal_loc[0], goal_loc[1])] = 'g{}'.format(i + 1)

        # Add objects with shapes and colors
        if 'objects' in self.env_settings:
            for (x, y), (shape, color) in self.env_settings['objects'].items():
                self.objects[(x, y)] = (shape, color)  # Store both shape and color


        # Handle the yellow button with initial shape and color
        yellow_button_coords = self.env_settings['yellow_button']
        for (x, y), (shape, color) in yellow_button_coords.items():
            self.objects[(x, y)] = (shape, color)  # Initially store yellow button with its shape and color

        self.p = self.env_settings['p']

        # Set the available actions of all agents
        actions = np.array([Actions.up.value, Actions.right.value, Actions.left.value, Actions.down.value, Actions.change_color_down.value, Actions.change_color_up.value, Actions.change_shape_down.value, Actions.change_shape_up.value], dtype=int)
        self.actions = np.full((self.num_agents, len(actions)), -2, dtype=int)
        for i in range(self.num_agents):
            self.actions[i] = actions

        # Assuming you have shape_mapping and color_mapping dictionaries
        

        # Total possible states considering each cell can have any shape and color
        self.num_states = self.Nr * self.Nc * num_shapes * num_colors


        
        # Define forbidden transitions corresponding to map edges
        self.forbidden_transitions = set()
        
        for row in range(self.Nr):
            self.forbidden_transitions.add((row, 0, Actions.left)) # If in left-most column, can't move left.
            self.forbidden_transitions.add((row, self.Nc - 1, Actions.right)) # If in right-most column, can't move right.
        for col in range(self.Nc):
            self.forbidden_transitions.add((0, col, Actions.up)) # If in top row, can't move up
            self.forbidden_transitions.add((self.Nr - 1, col, Actions.down)) # If in bottom row, can't move down

    def environment_step(self, s, a):
        """
        Execute actions a from states s.

        Parameters
        ----------
        s : numpy array
            Array of integers representing the current environment states of the agents.
        a : numpy array
            Array of integers representing the actions being taken by the agents.

        Outputs
        -------
        r : float
            Total reward achieved by taking actions a from states s.
        l : list
            List of events occurring at this step.
        s_next : numpy array
            Array of indices of next states.
        """
        num_agents = len(s)
        s_next = np.zeros_like(s)
        last_actions = np.zeros_like(a)
        
        # Update self.objects_before before agents act
        self.objects_before = self.objects.copy()
        
        # For each agent, get the next state
        for i in range(num_agents):
            s_next[i], last_actions[i] = self.get_next_state(s[i], a[i])
        
        # Now, check for object property changes
        for i in range(num_agents):
            # Get the agent's position after the action
            row_agent, col_agent, agent_shape, agent_color = self.get_state_description(s_next[i])
            agent_action = last_actions[i]
            position = (row_agent, col_agent)
            
            # Check if the agent is at an object's position
            if position in self.objects:
                # Get the object's current properties
                obj_shape, obj_color = self.objects[position]
                
                # Agent 1 changes color
                if i == 0:
                    if agent_action == Actions.change_color_up.value:
                        if obj_color < 2:
                            obj_color += 1
                            self.objects[position] = (obj_shape, obj_color)
                    elif agent_action == Actions.change_color_down.value:
                        if obj_color > 0:
                            obj_color -= 1
                            self.objects[position] = (obj_shape, obj_color)
                
                # Agent 2 changes shape
                if i == 1:
                    if agent_action == Actions.change_shape_up.value:
                        if obj_shape < 2:
                            obj_shape += 1
                            self.objects[position] = (obj_shape, obj_color)
                    elif agent_action == Actions.change_shape_down.value:
                        if obj_shape > 0:
                            obj_shape -= 1
                            self.objects[position] = (obj_shape, obj_color)
        
        # Now get the labels
        l = self.get_mdp_label(s, s_next, self.u)
        r = 0

        for e in l:
            # Get the new reward machine state and the reward of this step
            u2 = self.reward_machine.get_next_state(self.u, e)
            r = r + self.reward_machine.get_reward(self.u, u2)
            # Update the reward machine state
            self.u = u2

        return r, l, s_next


    def get_next_state(self, s, a):
        """
        Get the next state in the environment given action a is taken from state s.
        Update the last action that was truly taken due to MDP slip.

        Parameters
        ----------
        s : int
            Index of the current state.
        a : int
            Action to be taken from state s.

        Outputs
        -------
        s_next : int
            Index of the next state.
        last_action : int
            Last action the agent truly took because of slip probability.
        """
        slip_p = [self.p, (1-self.p)/2, (1-self.p)/2]
        check = random.random()

        row, col, shape, color = self.get_state_description(s)


        if check <= slip_p[0]:
            a_ = a

        elif check > slip_p[0] and check <= (slip_p[0] + slip_p[1]):
            if a == 0:  # up
                a_ = 3  # left
            elif a == 2:  # down
                a_ = 1  # right
            elif a == 3:  # left
                a_ = 2  # down
            elif a == 1:  # right
                a_ = 0  # up
            elif a == 4:  # change_color_up
                a_ = 5  # change_color_down
            elif a == 5:  # change_color_down
                a_ = 4  # change_color_up
            elif a == 6:  # change_shape_up
                a_ = 7  # change_shape_down
            elif a == 7:  # change_shape_down
                a_ = 6  # change_shape_up

        else:
            if a == 0:  # up
                a_ = 1  # right
            elif a == 2:  # down
                a_ = 3  # left
            elif a == 3:  # left
                a_ = 0  # up
            elif a == 1:  # right
                a_ = 2  # down
            elif a == 4:  # change_color_up
                a_ = 6  # change_shape_up
            elif a == 5:  # change_color_down
                a_ = 7  # change_shape_down
            elif a == 6:  # change_shape_up
                a_ = 4  # change_color_up
            elif a == 7:  # change_shape_down
                a_ = 5  # change_color_down 

        action_ = Actions(a_)
        # print('action: {}'.format(action_))
        if (row, col, action_) not in self.forbidden_transitions:
            if action_ == Actions.up:
                row -= 1
            if action_ == Actions.down:
                row += 1
            if action_ == Actions.left:
                col -= 1
            if action_ == Actions.right:
                col += 1
            if action_ == Actions.change_color_down:
                if color > 0:
                    color -=1
            if action_ == Actions.change_color_up:
                if color < 2:
                    color += 1
            if action_ == Actions.change_shape_down:
                if shape > 0:
                    shape -= 1
            if action_ == Actions.change_shape_up:
                if shape < 2:
                    shape += 1

        s_next = self.get_state_from_description(row, col, shape, color)

        last_action = a_
        return s_next, last_action

    def get_state_from_description(self, row, col, shape, color):
        """
        Given a (row, column) index description of gridworld location, return
        index of corresponding state.

        Parameters
        ----------
        row : int
            Index corresponding to the row location of the state in the gridworld.
        col : int
            Index corresponding to the column location of the state in the gridworld.
        
        Outputs
        -------
        s : int
            The index of the gridworld state corresponding to location (row, col).
        """

        num_shapes = self.env_settings['length']
        num_colors = self.env_settings['length']


        position_index = self.Nc * row + col  # Base index for grid position
        s = (position_index * num_shapes * num_colors) + (shape * num_colors) + color

        return s
        # return self.Nc * row + col

    def get_state_description(self, s):
        """
        Return the row and column indeces of state s in the gridworld.

        Parameters
        ----------
        s : int
            Index of the gridworld state.

        Outputs
        -------
        row : int
            The row index of state s in the gridworld.
        col : int
            The column index of state s in the gridworld.
        """
        num_shapes = self.env_settings['length']
        num_colors = self.env_settings['length']

        base_index = self.Nr * self.Nc
    
        # Calculate the row and column from the base grid position
        position_index = np.floor_divide(s, (num_shapes * num_colors))
        row = np.floor_divide(position_index, self.Nc)
        col = np.mod(position_index, self.Nc)

        # Extract the shape and color information
        remaining_index = np.mod(s, (num_shapes * num_colors))
        shape = np.floor_divide(remaining_index, num_colors)
        color = np.mod(remaining_index, num_colors)

        return (row, col, int(shape), int(color))


        # row = np.floor_divide(s, self.Nr)
        # col = np.mod(s, self.Nc)

        # return (row, col)

    def get_actions(self, id):
        """
        Returns the list with the actions that a particular agent can perform.

        Parameters
        ----------
        id : int
            Index of the agent whose initial state is being queried.
        """
        return np.copy(self.actions[id])

    def get_last_action(self, id):
        """
        Returns a particular agent's last action.

        Parameters
        ----------
        id : int
            Index of the agent whose initial state is being queried.
        """
        return self.last_action[id]

    def get_team_action_array(self):
        """
        Returns the available actions of the entire team.

        Outputs
        -------
        actions : (num_agents x num_actions) numpy integer array
        """
        return np.copy(self.actions)
    
    def get_initial_state(self, id):
        """
        Returns the initial state of a particular agent.

        Parameters
        ----------
        id : int
            Index of the agent whose initial state is being queried.
        """
        return self.s_i[id]

    def get_initial_team_state(self):
        """
        Return the intial state of the collective multi-agent team.

        Outputs
        -------
        s_i : numpy integer array
            Array of initial state indices for the agents in the experiment.
        """
        return np.copy(self.s_i)

    ############## DQPRM-RELATED METHODS ########################################
    def get_mdp_label(self, s, s_next, u):
        """
        Get the MDP label resulting from transitioning from state s to state s_next.

        Parameters
        ----------
        s : numpy integer array
            Array of integers representing the environment states of the various agents.
        s_next : numpy integer array
            Array of integers representing the next environment states of the various agents.
        u : int
            Index of the reward machine state
        last_actions : numpy array
            Array of last actions taken by the agents.

        Outputs
        -------
        l : list of strings
            MDP labels resulting from the state transition.
        """

        l = []

        # Mappings from names to codes
        shape_mapping = {'circle': 0, 'square': 1, 'triangle': 2}
        color_mapping = {'red': 0, 'blue': 1, 'green': 2}

        # Get the positions of the objects
        object_positions = list(self.objects.keys())

        # Ensure the object positions are consistent with object IDs
        object_requirements = {
            1: {
                'position': object_positions[0],  # Position of Object 1
                'required_color': color_mapping['blue'],
                'required_shape': shape_mapping['square'],
                'label': 'a'
            },
            2: {
                'position': object_positions[1],  # Position of Object 2
                'required_color': color_mapping['red'],
                'required_shape': shape_mapping['circle'],
                'label': 'b'
            },
            3: {
                'position': object_positions[2],  # Position of Object 3
                'required_color': color_mapping['red'],
                'required_shape': shape_mapping['square'],
                'label': 'c'
            },
            4: {
                'position': object_positions[3],  # Position of Object 4
                'required_color': color_mapping['blue'],
                'required_shape': shape_mapping['triangle'],
                'label': 'd'
            },
        }

        # Initialize completed objects if not already done
        if not hasattr(self, 'completed_objects'):
            self.completed_objects = set()

        # Ensure self.objects_before is defined
        if not hasattr(self, 'objects_before'):
            self.objects_before = self.objects.copy()

        # Get Agent positions
        row_agent1, col_agent1, _, _ = self.get_state_description(s_next[0])
        row_agent2, col_agent2, _, _ = self.get_state_description(s_next[1])

        # Iterate over each object
        for obj_id, obj_info in object_requirements.items():
            position = obj_info['position']
            required_color_code = obj_info['required_color']
            required_shape_code = obj_info['required_shape']
            label = obj_info['label']

            # Get the object's properties before and after the transition
            # Assuming self.objects maps positions to (shape_code, color_code)
            shape_before, color_before = self.objects_before.get(position, (None, None))
            shape_after, color_after = self.objects.get(position, (None, None))

            # Flags to check if the required changes have been made
            color_changed_by_agent1 = False
            shape_changed_by_agent2 = False

            # Check if Agent 1 changed the object's color to the required color
            if color_before != required_color_code and color_after == required_color_code:
                # Verify that Agent 1 is at the object's position
                if (row_agent1, col_agent1) == position:
                    color_changed_by_agent1 = True

            # Check if Agent 2 changed the object's shape to the required shape
            if shape_before != required_shape_code and shape_after == required_shape_code:
                # Verify that Agent 2 is at the object's position
                if (row_agent2, col_agent2) == position:
                    shape_changed_by_agent2 = True

            # Debug print statements
            # print(f"\nObject {obj_id} at position {position}")
            # print(f"Before - Shape: {shape_before}, Color: {color_before}")
            # print(f"After - Shape: {shape_after}, Color: {color_after}")
            # print(f"Agent 1 Position: ({row_agent1}, {col_agent1})")
            # print(f"Agent 2 Position: ({row_agent2}, {col_agent2})")
            # print(f"Color changed by Agent 1: {color_changed_by_agent1}")
            # print(f"Shape changed by Agent 2: {shape_changed_by_agent2}")

            # If both changes have been made and the object is not yet completed
            if color_changed_by_agent1 and shape_changed_by_agent2 and obj_id not in self.completed_objects:
                print(f"Label '{label}' added for Object {obj_id}")
                l.append(label)
                self.completed_objects.add(obj_id)
            # else:
            #     print(f"No label added for Object {obj_id}")

        # Update self.objects_before for the next time step
        self.objects_before = self.objects.copy()

        return l








    ################## HRL-RELATED METHODS ######################################
    def get_options_list(self, agent_id):
        """
        Return a list of strings representing the possible options for each agent.

        Input
        -----
        agent_id : int
            The id of the agent whose option list is to be returned.
        
        Output
        ------
        options_list : list
            list of strings representing the options avaialble to the agent.
        """

        options_list = []
        options_list.append('r{}'.format(agent_id + 1)) # Navigate to rendezvous
        options_list.append('g{}'.format(agent_id + 1)) # Navigate to goal

        return options_list

    def get_avail_options(self, agent_id):
        """
        Given the current metastate, get the available options. Some options are unavailable if 
        they are not possible to complete at the current stage of the task. In such circumstances
        we don't want the agents to update the corresponding option q-functions.
        """
        avail_options = self.get_options_list(agent_id)

        return avail_options

    def get_avail_meta_action_indeces(self, agent_id):
        """
        Get a list of the indeces corresponding to the currently available meta-action/option
        """
        avail_options = self.get_avail_options(agent_id)
        all_options_list = self.get_options_list(agent_id)
        avail_meta_action_indeces = []
        for option in avail_options:
            avail_meta_action_indeces.append(all_options_list.index(option))
        return avail_meta_action_indeces

    def get_completed_options(self, s):
        """
        Get a list of strings corresponding to options that are deemed complete in the team state described by s.

        Parameters
        ----------
        s : numpy integer array
            Array of integers representing the environment states of the various agents.
            s[id] represents the state of the agent indexed by index "id".

        Outputs
        -------
        completed_options : list
            list of strings corresponding to the completed options.
        """
        completed_options = []

        for i in range(self.num_agents):
            row, col = self.get_state_description(s[i])
            if (row, col) in self.objects:
                if self.objects[(row, col)] == "w":
                    completed_options.append('r{}'.format(i+1))
                elif self.objects[(row, col)] == "g{}".format(i+1):
                    completed_options.append('g{}'.format(i+1))
        
        return completed_options

    def get_num_meta_states(self, agent_id):
        """
        Return the number of meta states for the agent specified by agent_id.
        """
        return int(3)
        # return len(self.reward_machine.U)

    def get_meta_state(self, agent_id):
        """
        Return the meta-state that the agent should use for it's meta controller.

        Input
        -----
        s_team : numpy array
            s_team[i] is the state of agent i.
        agent_id : int
            Index of agent whose meta-state is to be returned.

        Output
        ------
        meta_state : int
            Index of the meta-state.
        """
        if not (self.rendezvous_complete):
            return 0
        elif self.rendezvous_complete and not(self.reached_goal_flag[agent_id]):
            return 1
        elif self.rendezvous_complete and self.reached_goal_flag[agent_id]:
            return 2

        # Suprisingly, directly using the states of the reward machine as the meta states 
        # did not work well.
        # return self.u

    ######################### TROUBLESHOOTING METHODS ################################

    def show(self, s):
        """
        Create a visual representation of the current state of the gridworld.

        Parameters
        ----------
        s : int
            Index of the current state
        """
        display = np.zeros((self.Nr, self.Nc))

        # Display the location of key points in world
        for loc in self.objects.keys():
            display[loc] = 9

        # Display the agents
        for i in range(self.num_agents):
            row, col = self.get_state_description(s[i])
            display[row, col] = i + 1

        print(display)

def play():
    n = 2 # num agents

    # Set the environment settings for the experiment
    env_settings = dict()
    env_settings['Nr'] = 10
    env_settings['Nc'] = 10
    env_settings['initial_states'] = [0, 3, 20, 8, 90, 40, 70, 49, 96, 69]
    env_settings['rendezvous_loc'] = (3,4)
    env_settings['goal_locations'] = [(9,7), (7,9), (2,9), (9,9), (0,9), (7,0), (4,0), (5,0), (6,9), (8,0)]
    env_settings['p'] = 1.0

    base_file_dir = os.path.abspath(os.path.join(os.getcwd(), '../../..'))
    rm_string = os.path.join(base_file_dir, 'experiments', 'gridworld_many_agent_rendezvous', '{}_agent_rendezvous_rm.txt'.format(n))

    game = MultiAgentGridWorldEnv(rm_string, n, env_settings)

    # User inputs
    str_to_action = {"w":Actions.up.value,"d":Actions.right.value,"s":Actions.down.value,"a":Actions.left.value,"x":Actions.none.value}

    s = game.get_initial_team_state()
    print(s)

    while True:
        # Showing game
        game.show(s)

        # Getting action
        a = np.full(n, -1, dtype=int)
        
        for i in range(n):
            print('\nAction{}?'.format(i+1), end='')
            usr_inp = input()
            print()

            if not(usr_inp in str_to_action):
                print('forbidden action')
                a[i] = str_to_action['x']
            else:
                print(str_to_action[usr_inp])
                a[i] = str_to_action[usr_inp]

        r, l, s = game.environment_step(s, a)
        
        print("---------------------")
        print("Next States: ", s)
        print("Label: ", l)
        print("Reward: ", r)
        print("RM state: ", game.u)
        print("meta state: ", game.get_meta_state(0))
        print("---------------------")

        if game.reward_machine.is_terminal_state(game.u): # Game Over
                break 
    game.show(s)
    
# This code allow to play a game (for debugging purposes)
if __name__ == '__main__':
    play()