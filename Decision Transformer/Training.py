from cgi import test
from itertools import accumulate
import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score
import tensorflow as tf
import tqdm

from DecisionTransformer import *

capacity = 80000
episode_len = 10
field_size = 6

batch_size = 32

def loadData():

    global capacity, episode_len

    buff_states = np.load(f"./TrainingData/states_{capacity}.npy")
    buff_actions = np.load(f"./TrainingData/actions_{capacity}.npy")
    buff_rewards = np.load(f"./TrainingData/rewards_{capacity}.npy")

    for buff_idx in range(capacity):

        for episode_idx in range(episode_len):
            states = buff_states[buff_idx]
            states[episode_idx:episode_len-1, :, :] = 0

            actions = buff_actions[buff_idx]
            actions[episode_idx:episode_len-1] = 0

            rewards = buff_rewards[buff_idx]
            rewards[episode_idx:episode_len-1] = 0

            yield states, actions, rewards
        #yield (states[buff_idx], actions[buff_idx], rewards[buff_idx])


def main():

    global episode_len
    global field_size

    # Logging
    file_path = "test_logs/test" 
    train_summary_writer = tf.summary.create_file_writer(file_path)

    dataset = tf.data.Dataset.from_generator(loadData, 
        output_signature=(
            tf.TensorSpec(shape=(episode_len, field_size, field_size), dtype=tf.uint8), # states
            tf.TensorSpec(shape=(episode_len,), dtype=tf.uint8),  # actions
            tf.TensorSpec(shape=(episode_len,), dtype=tf.float32) # rewards
        )
    )

    train_size = 700000

    train_dataset = dataset.take(train_size)
    train_dataset = train_dataset.apply(prepare_data)

    test_dataset = dataset.take(1000)
    test_dataset = test_dataset.apply(prepare_data)
 

    num_epochs = 20

    decisionTransformer = DecisionTransformer(episode_len,num_actions=field_size*field_size*4)
  
    log(train_summary_writer, decisionTransformer, train_dataset, test_dataset, epoch=0)

    for epoch in range(num_epochs):
            
        print(f"Epoch {epoch}")

        for game_states, actions_target, rewards in tqdm.tqdm(train_dataset, total=int(train_size/batch_size)): 
            decisionTransformer.train_step(game_states, rewards, actions_target)

        log(train_summary_writer, decisionTransformer, train_dataset, test_dataset, epoch + 1)
        

def prepare_data(data):
    
    global episode_len
    global field_size
    global batch_size 

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
    data = data.batch(batch_size)
    data = data.prefetch(tf.data.experimental.AUTOTUNE)

    return data


def log(train_summary_writer, decisionTransformer, train_dataset, test_dataset, epoch):

    with train_summary_writer.as_default():
        
        if epoch == 0:
            decisionTransformer.test_step(train_dataset.take(1000))

   
        loss = decisionTransformer.metric_loss.result()
        tf.summary.scalar(f"train_loss", loss, step=epoch)
        
        accuracy = decisionTransformer.metric_accuracy.result()
        tf.summary.scalar(f"train_accuracy", accuracy, step=epoch)

        print(f"train_loss: {loss}")
        print(f"train_accuracy: {accuracy}")

        decisionTransformer.metric_loss.reset_states()
        decisionTransformer.metric_accuracy.reset_states()


        decisionTransformer.test_step(test_dataset)

        loss = decisionTransformer.metric_loss.result()
        tf.summary.scalar(f"test_loss", loss, step=epoch)

        accuracy = decisionTransformer.metric_accuracy.result()
        tf.summary.scalar(f"test_accuracy", accuracy, step=epoch)

        print(f"test_loss: {loss}")
        print(f"test_accuracy: {accuracy}")

        decisionTransformer.metric_loss.reset_states()
        decisionTransformer.metric_accuracy.reset_states()

        # if epoch == 0:
        #     train_dataset = train_dataset.take(500) # approx full train dataset
        #     mean_loss = decisionTransformer.test(train_dataset)
        #     tf.summary.scalar(f"train_loss", mean_loss, step=0)
        
        # else:
        #     mean_loss = decisionTransformer.metric_loss.result()
        #     tf.summary.scalar(f"train_loss", mean_loss, step=epoch)

        # mean_loss = decisionTransformer.test(test_dataset)
        # tf.summary.scalar(f"test_loss", mean_loss, step=epoch)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")