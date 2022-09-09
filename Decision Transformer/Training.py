import numpy as np
import tensorflow as tf
import tqdm

from DecisionTransformer import *

capacity = 40000
episode_len = 10
field_size = 6

def loadData():

    global capacity

    states = np.load(f"./TrainingData/states_{capacity}.npy")
    actions = np.load(f"./TrainingData/actions_{capacity}.npy")
    rewards = np.load(f"./TrainingData/rewards_{capacity}.npy")

    for buff_idx in range(capacity):
        yield (states[buff_idx], actions[buff_idx], rewards[buff_idx])


def main():

    global episode_len
    global field_size

    dataset = tf.data.Dataset.from_generator(loadData, 
        output_signature=(
            tf.TensorSpec(shape=(episode_len, field_size, field_size), dtype=tf.uint8), # states
            tf.TensorSpec(shape=(episode_len,), dtype=tf.uint8),  # actions
            tf.TensorSpec(shape=(episode_len,), dtype=tf.float32) # rewards
        )
    )

    dataset = dataset.apply(prepare_data)

    train_dataset = dataset.take(100)
    test_dataset = dataset.take(100)

    num_epochs = 1

    decisionTransformer = DecisionTransformer(episode_len,num_actions=field_size*field_size*4)

    for epoch in range(num_epochs):
            
        print(f"Epoch {epoch}")

        for states, actions, rewards in tqdm.tqdm(train_dataset,position=0, leave=True): 
            decisionTransformer(states)
            
    # for data in dataset.take(1):
    #     states, actions, rewards = data
    #     print(states)
    #     print(actions)
    #     print(rewards)


def prepare_data(data):
    
    global episode_len
    global field_size

    #
    # onehotify states
    #

    # state shape: (episode_len, field_size, field_size) -> (episode_len * field_size * field_size)
    data = data.map(lambda states, actions, rewards: (tf.reshape(states, shape=(episode_len*field_size*field_size,)), actions, rewards))

    # one hot
    num_one_hot = 26 # num of candys
    data = data.map(lambda states, actions, rewards: (tf.one_hot(states, depth=num_one_hot), actions, rewards))

    # state shape: (episode_len * field_size * field_size) -> (episode_len, field_size, field_size)
    data = data.map(lambda states, actions, rewards: (tf.reshape(states, shape=(episode_len, field_size, field_size, num_one_hot)), actions, rewards))


    #
    # onehotify actions
    #

    num_actions = field_size*field_size*4
    data = data.map(lambda states, actions, rewards: (states, tf.one_hot(actions, depth=num_actions), rewards))

    #
    # cache, shuffle, batch, prefetch
    # 
    data = data.cache()
 
    data = data.shuffle(1000)
    data = data.batch(16)
    data = data.prefetch(tf.data.experimental.AUTOTUNE)

    return data

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")