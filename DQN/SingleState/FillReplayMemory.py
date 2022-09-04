import sys
# sys.path.append("../..")
sys.path.append("../AlphaCandy")
from CandyCrushGym import *

import tqdm
import os

import gym

def sample_actions(num_actions, field_size=8):

    actions = []
    for i in range(num_actions):

        while True:
            action = np.random.randint(0, 255)

            fieldID = action // 4

            direction = action % 4

            x = fieldID // field_size
            y = fieldID % field_size

            # Swap candy
            x_swap = x # attention: numpy x->y are swapped
            y_swap = y # attention: numpy x->y are swapped
            # top
            if direction == 0:
                y_swap += -1
            # down
            elif direction == 2: 
                y_swap += 1
            # right 
            elif direction == 1:
                x_swap += 1
            # left 
            elif direction == 3:
                x_swap += -1

            if isValidIndex(x,y, field_size) and isValidIndex(x_swap, y_swap, field_size):
                break
        
    
        actions.append(action)

    return np.array(actions)

def isValidIndex(x, y, field_size=8):
        if field_size <= x or field_size <= y or x < 0 or y < 0:
            return False
        return True

def main(field_size=8, num_elements=6):
    
    batch_size = 16
    
    envs = gym.vector.AsyncVectorEnv([
            lambda: CandyCrushGym(100, field_size, num_elements),
            lambda: CandyCrushGym(200, field_size, num_elements),
            lambda: CandyCrushGym(300, field_size, num_elements),
            lambda: CandyCrushGym(400, field_size, num_elements),
            lambda: CandyCrushGym(500, field_size, num_elements),
            lambda: CandyCrushGym(600, field_size, num_elements),
            lambda: CandyCrushGym(700, field_size, num_elements),
            lambda: CandyCrushGym(800, field_size, num_elements),
            lambda: CandyCrushGym(900, field_size, num_elements),
            lambda: CandyCrushGym(1000, field_size, num_elements),
            lambda: CandyCrushGym(1100, field_size, num_elements),
            lambda: CandyCrushGym(1200, field_size, num_elements),
            lambda: CandyCrushGym(1300, field_size, num_elements),
            lambda: CandyCrushGym(1400, field_size, num_elements),
            lambda: CandyCrushGym(1500, field_size, num_elements),
            lambda: CandyCrushGym(1600, field_size, num_elements),
        ])
    
    capacity = 2500000
    
    num_episodes = 1000
    episode_len = 50

    idx = 0

    field_shape = envs.observation_space.shape[1:] # ignore batch size
    buff_states = np.zeros((capacity, *field_shape), dtype=np.int8)
    buff_actions = np.zeros(capacity, dtype=np.int8)
    buff_next_states = np.zeros((capacity, *field_shape), dtype=np.int8)
    buff_rewards = np.zeros(capacity, dtype=np.float32)
    
    for _ in tqdm.tqdm(range(num_episodes), position=0, leave=True):
        states = envs.reset()
     
        for _ in range(episode_len):
            actions = sample_actions(batch_size, field_size)

            next_states, rewards, _, _ = envs.step(actions)

            for i in range(batch_size):

                if rewards[i] == 0:
                    x = np.random.randint(0, 10)
                    if x == 0:

                        buff_states[idx] = states[i]
                        buff_actions[idx] = actions[i]
                        buff_next_states[idx] = next_states[i]
                        buff_rewards[idx] = rewards[i]

                        idx+=1
                else:
                    buff_states[idx] = states[i]
                    buff_actions[idx] = actions[i]
                    buff_next_states[idx] = next_states[i]
                    buff_rewards[idx] = rewards[i]

                    idx+=1
            
            states = next_states

    filename = "./ReplayMemoryData/"
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    np.save(f"./ReplayMemoryData/states_{idx}", buff_states)
    np.save(f"./ReplayMemoryData/actions_{idx}", buff_actions)
    np.save(f"./ReplayMemoryData/next_states_{idx}", buff_next_states)
    np.save(f"./ReplayMemoryData/rewards_{idx}", buff_rewards)


if __name__ == "__main__":
    try:
        main(field_size=8, num_elements=6)
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")