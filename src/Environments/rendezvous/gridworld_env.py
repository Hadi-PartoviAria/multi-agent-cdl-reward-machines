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

class GridWorldEnv:

    def __init__(self, rm_file, agent_id, env_settings):
        """
        Initialize gridworld environment.

        Parameters
        ----------
        rm_file : string
            File path leading to the text file containing the reward machine
            encoding this environment's reward function.
        agent_id : int
            Index {0,1} indicating which agent
        env_settings : dict
            Dictionary of environment settings
        """
        self.env_settings = env_settings
        self.agent_id = agent_id
        self._load_map()
        self.reward_machine = SparseRewardMachine(rm_file)

        self.u = self.reward_machine.get_initial_state()
        self.last_action = -1 # Initialize last action to garbage value

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
                'required_color': color_mapping['green'],
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

    def get_actions(self):
        """
        Returns the list with the actions that the agent can perform
        """
        return self.actions

    def get_last_action(self):
        """
        Returns agent's last action
        """
        return self.last_action

    def get_initial_state(self):
        """
        Outputs
        -------
        s_i : int
            Index of agent's initial state.
        """
        return self.s_i
