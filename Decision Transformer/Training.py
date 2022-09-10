from calendar import day_abbr
import numpy as np
import tensorflow as tf
import tqdm

from DecisionTransformer import *

capacity = 20000
episode_len = 10
field_size = 6

batch_size = 32

def loadData():

    global capacity, episode_len, field_size

    none_action_id = field_size*field_size*4 + 1

    buff_states = np.load(f"./TrainingData/states_{capacity}.npy")
    buff_rewards = np.load(f"./TrainingData/rewards_{capacity}.npy")
    buff_actions = np.load(f"./TrainingData/actions_{capacity}.npy")

    for buff_idx in range(capacity):

        for episode_idx in range(episode_len):
            states = np.copy(buff_states[buff_idx])
            states[episode_idx + 1:episode_len, :, :] = 0

            actions = np.copy(buff_actions[buff_idx])
            actions[episode_idx:episode_len] = none_action_id

            rewards = np.copy(buff_rewards[buff_idx])
            rewards[episode_idx + 1:episode_len] = 0

            action_target = buff_actions[buff_idx, episode_idx]
          
            yield states, actions, rewards, action_target
     


def main():

    global episode_len
    global field_size

    num_epochs = 20
    train_size = 20000


    # Logging
    file_path = "test_logs/test" 
    train_summary_writer = tf.summary.create_file_writer(file_path)

    dataset = tf.data.Dataset.from_generator(loadData, 
        output_signature=(
            tf.TensorSpec(shape=(episode_len, field_size, field_size), dtype=tf.uint8), # states
            tf.TensorSpec(shape=(episode_len,), dtype=tf.uint8),  # actions
            tf.TensorSpec(shape=(episode_len,), dtype=tf.float32), # rewards
            tf.TensorSpec(shape=(), dtype=tf.uint8),  # action_target
        )
    )

    #
    # Dataset
    #
    train_dataset = dataset.take(train_size)
    train_dataset = train_dataset.apply(prepare_data)

    test_dataset = dataset.take(1000)
    test_dataset = test_dataset.apply(prepare_data)
    
    decisionTransformer = DecisionTransformer(episode_len,num_actions=field_size*field_size*4)
    
    log(train_summary_writer, decisionTransformer, train_dataset, test_dataset, epoch=0)

    #
    # Train loop
    #

    for epoch in range(num_epochs):
            
        print(f"Epoch {epoch}")

        for states, actions, rewards, action_target in tqdm.tqdm(train_dataset, total=int(train_size/batch_size)): 
            decisionTransformer.train_step(states, actions, rewards, action_target)

        log(train_summary_writer, decisionTransformer, train_dataset, test_dataset, epoch + 1)
        decisionTransformer.save_weights(f"./saved_models/trained_weights_epoch_{epoch + 1}", save_format="tf")

def prepare_data(data):
    
    global episode_len
    global field_size
    global batch_size 

    #
    # onehotify states
    #

    # state shape: (episode_len, field_size, field_size) -> (episode_len * field_size * field_size)
    data = data.map(lambda states, actions, rewards, action_target: (tf.reshape(states, shape=(episode_len*field_size*field_size,)), actions, rewards, action_target))

    # one hot
    num_one_hot = 26 # num of candys
    data = data.map(lambda states, actions, rewards, action_target: (tf.one_hot(states, depth=num_one_hot), actions, rewards, action_target))

    # state shape: (episode_len * field_size * field_size) -> (episode_len, field_size, field_size)
    data = data.map(lambda states, actions, rewards, action_target: (tf.reshape(states, shape=(episode_len, field_size, field_size, num_one_hot)), actions, rewards, action_target))


    #
    # onehotify actions
    #

    num_actions = field_size*field_size*4 + 1
    data = data.map(lambda states, actions, rewards, action_target: (states, tf.one_hot(actions, depth=num_actions), rewards, action_target))

    #
    # onehotify action_target
    #

    data = data.map(lambda states, actions, rewards, action_target: (states, actions, rewards, tf.one_hot(action_target, depth=num_actions)))

    #
    # cache, shuffle, batch, prefetch
    # 
    #data = data.cache()
 
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


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")