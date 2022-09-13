import tensorflow as tf

class DecisionTransformer(tf.keras.Model):
    def __init__(self, episode_len, num_actions):
        super(DecisionTransformer, self).__init__()

        self.episode_len = episode_len
        self.num_actions = num_actions + 1 # do not forget "none" action

        # state
        self.state_feature_extractor = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), activation="tanh", padding='same')
        self.state_bottleneck = tf.keras.layers.Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), activation="tanh", padding='same')
        self.state_embedding = tf.keras.layers.Dense(10, activation=tf.nn.tanh)

        # action
        self.action_embedding = tf.keras.layers.Dense(10, activation=tf.nn.tanh)

        self.self_attention = tf.keras.layers.MultiHeadAttention(num_heads=5, key_dim=20, value_dim=10)
        self.backend_layer_list = [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(100, activation=tf.nn.tanh),
            tf.keras.layers.Dense(self.num_actions, activation=tf.nn.softmax)
        ]

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.loss_function = tf.keras.losses.BinaryCrossentropy()

        self.metric_loss = tf.keras.metrics.Mean(name="loss")
        self.metric_accuracy = tf.keras.metrics.Accuracy(name="accuracy")


    @tf.function
    def call(self, states, actions, rewards):

        batch_size = states.shape[0]
        field_size = states.shape[2]

        # Extract features from each game state
        states_embeddings_list = []
        
        action_embeddings_list = []

        for episode_idx in range(self.episode_len):
            
            state = states[:, episode_idx, :, :]

            # feature_extractor -> bottleneck -> embedding
            state_embedding = self.state_feature_extractor(state)
            state_embedding = self.state_bottleneck(state_embedding)

            feature_map_size = state_embedding.shape[1]

            state_embedding = self.state_embedding(state_embedding)

            # embedding: flatten
            state_embedding = tf.reshape(state_embedding, shape=(batch_size, feature_map_size*feature_map_size*10))

            # collect embedding for each time step
            states_embeddings_list.append(state_embedding)


            action = actions[:, episode_idx]
            action_embedding = self.action_embedding(action)
            action_embeddings_list.append(action_embedding)
            

        # Merge embedding for each time step
        state_embeddings = tf.stack(states_embeddings_list,axis=1)
        action_embeddings = tf.stack(action_embeddings_list, axis=1)

        #
        # Merge embedding of game state with rewards
        #
        rewards = tf.expand_dims(rewards, axis=-1)
    
        input = tf.concat([state_embeddings, rewards, action_embeddings], axis=-1)
        
        x = self.self_attention(input, value=input)

        #
        # pass through backend
        #

        for layer in self.backend_layer_list:
            x = layer(x)

          
        return x

    @tf.function
    def train_step(self, states, actions, rewards, action_target):

        with tf.GradientTape() as tape:
            action_prediction = self(states, actions, rewards)

            
            loss = self.loss_function(action_target, action_prediction)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.metric_loss.update_state(loss)

        action_prediction = tf.argmax(action_prediction, axis=-1)
        action_target = tf.argmax(action_target, axis=-1)
        self.metric_accuracy.update_state(action_target, action_prediction)
    
    #@tf.function
    def test_step(self, dataset):
        
        self.metric_loss.reset_states()
        self.metric_accuracy.reset_states()

        for states, actions, rewards, action_target in dataset:
            action_prediction = self(states, actions, rewards)

            loss = self.loss_function(action_target, action_prediction)
            self.metric_loss.update_state(loss)

            action_prediction = tf.argmax(action_prediction, axis=-1)
            action_target = tf.argmax(action_target, axis=-1)
            self.metric_accuracy.update_state(action_target, action_prediction)