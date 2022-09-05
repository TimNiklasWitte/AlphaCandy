import sys
sys.path.append("../..")
sys.path.append("../AlphaCandy")
from CandyCrushGym import *

import tqdm
import os
import argparse
import gym

def sample_actions(num_actions, field_size):

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

def isValidIndex(x, y, field_size):
        if field_size <= x or field_size <= y or x < 0 or y < 0:
            return False
        return True


def checkFieldSize(size: str):
    size = int(size)
    if size <= 4:
        raise argparse.ArgumentTypeError("Field size must be greater than 4")
    return size

def checkNumCandys(num: int):
    num = int(num)
    if num <= 4:
        raise argparse.ArgumentTypeError("Number of candys must be greater than 4")
    return num

def main():
    
    # Set up ArgumentParser
    parser = argparse.ArgumentParser(description="Create inital data for the ReplayMemory.")
    parser.add_argument("--size", help="Set the field size.", type=checkFieldSize, required=True)
    parser.add_argument("--num", help="Set the number of candys.", type=checkNumCandys, required=True)

    args = parser.parse_args()

    field_size = args.size
    num_elements = args.num

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
    
    capacity = 1000000
    num_init_samples = 500000
    episode_len = 100

    field_shape = envs.observation_space.shape[1:] # ignore batch size
    buff_states = np.zeros((capacity, *field_shape), dtype=np.int8)
    buff_actions = np.zeros(capacity, dtype=np.int8)
    buff_next_states = np.zeros((capacity, *field_shape), dtype=np.int8)
    buff_rewards = np.zeros(capacity, dtype=np.float32)
    
    idx = 0
    while idx < num_init_samples:
        states = envs.reset()

        print(idx)
        
        for _ in range(episode_len):

            if idx == num_init_samples:
                break

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
                
                if idx == num_init_samples:
                    break
                
            states = next_states

    filename = "./ReplayMemoryData/"
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    np.save(f"./ReplayMemoryData/states_{idx}", buff_states)
    np.save(f"./ReplayMemoryData/actions_{idx}", buff_actions)
    np.save(f"./ReplayMemoryData/next_states_{idx}", buff_next_states)
    np.save(f"./ReplayMemoryData/rewards_{idx}", buff_rewards)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")