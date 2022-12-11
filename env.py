import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as Image
import random
import pickle
import os
from typing import Tuple, List


class BaseGridEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
        self,
        size: Tuple[int, int],
        start: int,
        epsilon: float,
        obstacle: List[int],
        semi_obstacle: List[int],
    ):
      
        self.grid_map_shape = [size[0], size[1]]
        self.epsilon_original = epsilon
        self.epsilon = epsilon
        self.obstacles = obstacle
        self.semi_obstacles = semi_obstacle

        self.observation_space = spaces.Dict(
            {
                "position": spaces.Discrete(size[0] * size[1]),
                "visited": spaces.MultiBinary(size[0] * size[1]),
                "path": spaces.MultiBinary(size[0] * size[1]),
                "semi_obstacles": spaces.MultiBinary(size[0] * size[1]),
                "obstacles": spaces.MultiBinary(size[0] * size[1]),
            }
        )
        self.action_space = spaces.Discrete(4)
        visited = np.zeros(size[0] * size[1], dtype=bool)
        self.state_obstacles = np.zeros(size[0] * size[1], dtype=bool)
        self.state_semi_obstacles = np.zeros(size[0] * size[1], dtype=bool)

        for i in obstacle:
            visited[i] = True
            self.state_obstacles[i] = True

        for i in semi_obstacle:
            self.state_semi_obstacles[i] = True

        path = np.zeros(size[0] * size[1], dtype=bool)
        path[start] = True

        self.start_state = {
            "position": start if start is not None else 0,
            "visited": visited,
            "path": path,
            "semi_obstacles": self.state_semi_obstacles,
            "obstacles": self.state_obstacles,
        }
        self.visual_saved = {}

    def serial_to_twod(self, ind):
        """Convert a serialized state number to a 2D map's state coordinate"""
        return np.array([ind // self.grid_map_shape[1], ind % self.grid_map_shape[1]])

    def twod_to_serial(self, twod):
        """Convert a 2D map's state coordinate to a serialized state number"""
        return twod[0] * self.grid_map_shape[1] + twod[1]

    def reset(self):
        """Rest the environment by initializaing the start state"""
        self.observation = self.start_state
        return self.observation

    def render(self, mode="rgb_array", close=False):
        """Render the agent state"""
        pixel_size = 20
        img = np.zeros(
            [
                pixel_size * self.grid_map_shape[0],
                pixel_size * self.grid_map_shape[1],
                3,
            ],
            dtype=np.uint8,
        )

        visited = self.observation["visited"]
        for i in range(self.grid_map_shape[0] * self.grid_map_shape[1]):
            if visited[i]:
                pos_x, pos_y = self.serial_to_twod(i)
                img[
                    pixel_size * pos_x : pixel_size * (1 + pos_x),
                    pixel_size * pos_y : pixel_size * (1 + pos_y),
                ] = [255, 255, 150]

        for semi_obstacle in self.semi_obstacles:
            pos_x, pos_y = self.serial_to_twod(semi_obstacle)
            if visited[semi_obstacle]:
                img[
                    pixel_size * pos_x : pixel_size * (1 + pos_x),
                    pixel_size * pos_y : pixel_size * (1 + pos_y),
                ] = [170, 170, 120]
            else:
                img[
                    pixel_size * pos_x : pixel_size * (1 + pos_x),
                    pixel_size * pos_y : pixel_size * (1 + pos_y),
                ] = [150, 150, 150]

        for obstacle in self.obstacles:
            pos_x, pos_y = self.serial_to_twod(obstacle)
            img[
                pixel_size * pos_x : pixel_size * (1 + pos_x),
                pixel_size * pos_y : pixel_size * (1 + pos_y),
            ] = [0, 0, 0]

        agent_state = self.serial_to_twod(self.observation["position"])
        img[
            pixel_size * agent_state[0] : pixel_size * (1 + agent_state[0]),
            pixel_size * agent_state[1] : pixel_size * (1 + agent_state[1]),
        ] = [0, 0, 255]

        if mode == "human":
            fig = plt.figure(0)
            plt.clf()
            plt.imshow(img, cmap="gray")
            fig.canvas.draw()
            plt.pause(0.01)
        if mode == "rgb_array":
            return img

    def _close_env(self):
        """Close the environment screen"""
        plt.close(1)
        return


class GridEnv(BaseGridEnv):
    """
    A grid-world environment.
    """

    def load_visual(self):
        if os.path.isfile("visual_saved.pkl"):
            with open("visual_saved.pkl", "rb") as f:
                self.visual_saved = pickle.load(f)
        else:
            self.visual_saved = {}

    def load_visual_txt(self):
        if os.path.isfile("visual_saved.txt"):
            with open("visual_saved.txt", "r") as f:
                self.visual_saved = {}
                for i, data in enumerate(f.readlines()):
                    self.visual_saved[i] = np.array(
                        list(map(lambda x: x == "1", data.strip()))
                    )
        else:
            self.visual_saved = {}

    def save_visual(self):
        with open("data/visual_saved.pkl", "wb") as f:
            pickle.dump(self.visual_saved, f)

    def save_visual_txt(self):
        with open("data/visual_saved.txt", "w") as f:
            for i in range(self.grid_map_shape[0] * self.grid_map_shape[1]):
                if i in self.visual_saved:
                    f.write(
                        "".join(
                            map(
                                lambda x: "1" if x else "0",
                                self.visual_saved[i].tolist(),
                            )
                        )
                    )
                    f.write("\n")

    def save_visual_fancy(self):
        with open("data/visual_saved_fancy.txt", "w") as f:
            for pos in range(self.grid_map_shape[0] * self.grid_map_shape[1]):
                if pos in self.visual_saved:
                    for i in range(self.grid_map_shape[0]):
                        for j in range(self.grid_map_shape[1]):
                            x = i * self.grid_map_shape[1] + j
                            if pos == x:
                                f.write("O")
                            elif x in self.obstacles:
                                f.write("#")
                            elif x in self.semi_obstacles:
                                if self.visual_saved[pos][x]:
                                    f.write("+")
                                else:
                                    f.write("-")
                            elif self.visual_saved[pos][x]:
                                f.write(".")
                            else:
                                f.write(" ")
                        f.write("\n")
                    f.write("\n")
                    f.write("\n")

    def calc_visual(self):
        self.load_visual_txt()
        for i in range(self.grid_map_shape[0] * self.grid_map_shape[1]):
            b = i in self.visual_saved
            visual = self.check_visual(i)
            self.print_state({"position": i, "visited": visual})
            print(f"calc visual: {i}")
            if not b:
                self.save_visual()
                self.save_visual_txt()
                self.save_visual_fancy()
        self.save_visual()
        self.save_visual_txt()
        self.save_visual_fancy()

    def check_visual(self, position):
        if len(self.visual_saved) == 0:
            self.load_visual_txt()
        if position in self.visual_saved:
            return self.visual_saved[position]
        position_serial = position
        position = self.serial_to_twod(position)
        visited = np.zeros(self.grid_map_shape[0] * self.grid_map_shape[1], dtype=bool)
        ar = np.array

        def is_line_overlap(line1, line2):
            def cross(a, b):
                return a[0] * b[1] - a[1] * b[0]

            def sgn(x):
                if x < 0:
                    return -1
                if x == 0:
                    return 0
                return 1

            def ccw(a, o, b):
                return sgn(cross(a - o, b - o))

            a1 = line1[0]
            b1 = line1[1]
            a2 = line2[0]
            b2 = line2[1]

            return (
                ccw(a1, b1, a2) * ccw(a1, b1, b2) <= 0
                and ccw(a2, b2, a1) * ccw(a2, b2, b1) <= 0
            )

        obs_line = set()
        for i in self.obstacles:
            pos = self.serial_to_twod(i)
            x = pos[0]
            y = pos[1]
            a = ((x - 0.5, y - 0.5), (x - 0.5, y + 0.5))
            if a in obs_line:
                obs_line.remove(a)
            else:
                obs_line.add(a)
            a = ((x + 0.5, y - 0.5), (x + 0.5, y + 0.5))
            if a in obs_line:
                obs_line.remove(a)
            else:
                obs_line.add(a)
            a = ((x - 0.5, y + 0.5), (x + 0.5, y + 0.5))
            if a in obs_line:
                obs_line.remove(a)
            else:
                obs_line.add(a)
            a = ((x + 0.5, y - 0.5), (x - 0.5, y - 0.5))
            if a in obs_line:
                obs_line.remove(a)
            else:
                obs_line.add(a)

        for i in range(self.grid_map_shape[0] * self.grid_map_shape[1]):
            pos = self.serial_to_twod(i)
            a = True
            for j in obs_line:
                a = a and not is_line_overlap((position, pos), (ar(j[0]), ar(j[1])))
            visited[i] = a

        self.visual_saved[position_serial] = visited
        return visited

    def print_state(self, state):
        position = state["position"]
        visited = state["visited"]
        path = state["path"]
        s = ""
        for i in range(self.grid_map_shape[0]):
            for j in range(self.grid_map_shape[1]):
                x = i * self.grid_map_shape[1] + j
                if position == x:
                    s += "O"
                elif x in self.obstacles:
                    s += "#"
                elif x in self.semi_obstacles:
                    if visited[x]:
                        s += "+"
                    else:
                        s += "-"
                elif path[x]:
                    s += "*"
                elif visited[x]:
                    s += "."
                else:
                    s += " "
            s += "\n"
        print(s)

    def transition_model(self, state, action):

        if not isinstance(state, dict):
            state = state.item()

        # assert (self.state_obstacles == state["obstacles"]).all()

        position_serial = state["position"]
        position = self.serial_to_twod(position_serial)
        visited = state["visited"]

        # the transition probabilities to the next states
        probs = {}
        visited_dict = {}

        # Left top is [0,0],
        action_pos = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])

        if position_serial in self.obstacles or np.all(visited):
            probs[position_serial] = 1.0
            visited_dict[position_serial] = visited
            return probs, visited_dict
        action_probs = np.ones(self.action_space.n) * self.epsilon / 4
        action_probs[action] = 1 - self.epsilon

        tot = 0.0
        cnt = 0
        for new_action, prob in enumerate(action_probs):
            next_state = position + action_pos[new_action]

            if (
                next_state[0] < 0
                or next_state[0] >= self.grid_map_shape[0]
                or next_state[1] < 0
                or next_state[1] >= self.grid_map_shape[1]
                or self.twod_to_serial(next_state) in self.obstacles
            ):
                next_state = position

            tot += prob
            cnt += 1

            next_position = self.twod_to_serial(next_state)
            if next_position in probs:
                probs[next_position] += prob
            else:
                probs[next_position] = prob
            if next_position not in visited_dict:
                visited_dict[next_position] = np.logical_or(
                    self.check_visual(next_position), visited
                )

        return probs, visited_dict

    def compute_reward(self, state, action, next_state):
    

        reward = 0
        visited = state["visited"]
        chk = np.all(visited)

        if chk:
            reward = (
                self.grid_map_shape[0]
                * self.grid_map_shape[1]
                * self.grid_map_shape[0]
                * self.grid_map_shape[1]
            )
            # print("all checked")
        elif next_state["position"] in self.semi_obstacles:
            reward = -100
        elif state["path"][next_state["position"]]:
            reward = -200
        elif next_state["position"] == state["position"]:
            reward = -400
        else:
            reward = -0.1
            next_visited = next_state["visited"].sum()
            curr_visited = state["visited"].sum()
            reward += next_visited * next_visited - curr_visited * curr_visited
            # print(reward)

        return reward

    def is_done(self, state, action, next_state):
        """
        Return True when the agent is in a terminal state or obstacles,
        otherwise return False

        Parameters
        ----------
        state: integer
            a serialized state index
        action: integer
            action index
        next_state: integer
            a serialized state index

        Returns
        -------
        done: Bool
            the result of termination or collision
        """

        done = np.all(next_state["visited"])
        if done:
            self.print_state(next_state)
        return done

    def step(self, action):
        """
        A step function that applies the input action to the environment.

        Parameters
        ----------
        action: integer
            action index

        Returns
        -------
        observation: integer
            the outcome of the given action (i.e., next state)... s' ~ T(s'|s,a)
        reward: float
            the reward that would get for ... r(s, a, s')
        done: Bool
            the result signal of termination or collision
        info: Dictionary
            Information dictionary containing miscellaneous information...
            (Do not need to implement info)

        """
        done = False
        action = int(action)

        probs, visited_dict = self.transition_model(self.observation, action)
        next_state = random.choices(
            list(probs.keys()), weights=list(probs.values()), k=1
        )[0]
        path = self.observation["path"].copy()
        path[next_state] = True
        self.agent_state = {
            "position": next_state,
            "visited": visited_dict[next_state],
            "path": path,
            "semi_obstacles": self.state_semi_obstacles,
            "obstacles": self.state_obstacles,
        }
        # self.print_state(self.agent_state)

        old_obs = self.observation
        self.observation = self.agent_state
        reward = self.compute_reward(old_obs, action, self.observation)
        done = self.is_done(old_obs, action, self.observation)
        self.epsilon *= self.epsilon_original
        return (self.observation, reward, done, {})
