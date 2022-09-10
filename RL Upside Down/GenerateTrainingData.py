import sys
sys.path.append("../")
sys.path.append("../AlphaCandy")
from CandyCrushGym import *

import os
import argparse

import numpy as np
from multiprocessing import Process, shared_memory

def sample_action(field_size):

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
            return action

def isValidIndex(x, y, field_size):
    if field_size <= x or field_size <= y or x < 0 or y < 0:
        return False
    return True
    



def fill_buff(num_fill_buff_threads, samples_per_thread, thread_id, 
    field_size, num_elements, 
    capacity, 
    states_shm, actions_shm, rewards_shm):

    states = np.ndarray((capacity, field_size, field_size), dtype=np.uint8, buffer=states_shm.buf)
    actions = np.ndarray((capacity), dtype=np.uint8, buffer=actions_shm.buf)
    rewards = np.ndarray((capacity), dtype=np.float32, buffer=rewards_shm.buf)

    env = CandyCrushGym(thread_id, field_size, num_elements)

    # Seed env and np
    #env.seed(thread_id)
    np.random.seed(thread_id)

    state = env.reset()

    for i in range(samples_per_thread):

        repeat = True

        while repeat:

            action = sample_action(field_size)
            next_state, reward, _, _ = env.step(action)

            if reward == 0:
                x = np.random.randint(0, 20)

                if x == 0:
                    repeat = False
                else:
                    repeat = True
                
            else:
                repeat = False

            idx = num_fill_buff_threads*i + thread_id

            states[idx, :, :] = state
            actions[idx] = action
            rewards[idx] = reward

            state = next_state

            if i % 100 == 0:
                state = env.reset()
    
        if thread_id == 0 and i % 100 == 0:
            print(i)
                    

def checkFieldSize(size: str):
    size = int(size)
    if size <= 4:
        raise argparse.ArgumentTypeError("Field size must be greater than 4")
    return size

def checkNumCandys(num: int):
    num = int(num)
    if num <= 3:
        raise argparse.ArgumentTypeError("Number of candys must be greater than 3")
    return num


def main():
    
    # Set up ArgumentParser
    parser = argparse.ArgumentParser(description="Create inital data for the ReplayMemory.")
    parser.add_argument("--size", help="Set the field size.", type=checkFieldSize, required=True)
    parser.add_argument("--num", help="Set the number of candys.", type=checkNumCandys, required=True)

    args = parser.parse_args()

    field_size = args.size
    num_elements = args.num

    
    num_fill_buff_threads = 12
    samples_per_thread = 30000

    capacity = num_fill_buff_threads * samples_per_thread
    fill_buff_threads = []

    #
    # Create shared memory
    #
    states_shm = shared_memory.SharedMemory(create=True, size=capacity * field_size*field_size * 4)
    actions_shm = shared_memory.SharedMemory(create=True, size=capacity * 4)
    rewards_shm = shared_memory.SharedMemory(create=True, size=capacity * 4)
       
    states = np.ndarray((capacity, field_size, field_size), dtype=np.uint8, buffer=states_shm.buf)
    actions = np.ndarray((capacity,), dtype=np.uint8, buffer=actions_shm.buf)
    rewards = np.ndarray((capacity,), dtype=np.float32, buffer=rewards_shm.buf)
  
    #
    # Run threads
    #
    for thread_idx in range(num_fill_buff_threads):
        process = Process(target=fill_buff, args=(num_fill_buff_threads, samples_per_thread, thread_idx, 
                                                field_size, num_elements,
                                                capacity, 
                                                states_shm, actions_shm, rewards_shm))

        process.start()
        fill_buff_threads.append(process)

    #
    # Join
    #
    for process in fill_buff_threads: 
        process.join()
    

    #
    # Save data
    #
    filename = "./TrainingData/"
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    np.save(f"./TrainingData/states_{capacity}", states)
    np.save(f"./TrainingData/actions_{capacity}", actions)
    np.save(f"./TrainingData/rewards_{capacity}", rewards)

    #
    # Free shared memory
    #
    states_shm.close()
    actions_shm.close()
    rewards_shm.close()

    states_shm.unlink()
    actions_shm.unlink()
    rewards_shm.unlink()

    
    
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")