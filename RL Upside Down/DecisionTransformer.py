import tensorflow as tf

class DecisionTransformer(tf.keras.Model):
    def __init__(self, num_actions):
        super(DecisionTransformer, self).__init__()

        self.num_actions = num_actions

        self.frontend_layer_list = [
            tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(2,2), activation="tanh", padding='same'),
            tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), activation="tanh", padding='valid'),
            #tf.keras.layers.Conv2D(filters=1024, kernel_size=(3,3), strides=(2,2), activation="tanh", padding='same'),
            tf.keras.layers.Flatten()
        ]

        self.state_embedding = tf.keras.layers.Dense(100, activation=tf.nn.tanh)

        self.backend_layer_list = [
            tf.keras.layers.Dense(500, activation=tf.nn.tanh),
            tf.keras.layers.Dense(self.num_actions, activation=tf.nn.softmax)
        ]

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.loss_function = tf.keras.losses.BinaryCrossentropy()

        self.metric_loss = tf.keras.metrics.Mean(name="loss")
        self.metric_accuracy = tf.keras.metrics.Accuracy(name="accuracy")


    @tf.function
    def call(self, states, rewards):

        batch_size = states.shape[0]

        # feature_extractor -> bottleneck -> embedding
        
        x = states 
        for layer in self.frontend_layer_list:
            x = layer(x) 

        state_embedding = self.state_embedding(x)
  
        #
        # Merge embedding of game state with rewards
        #
        rewards = tf.expand_dims(rewards, axis=-1)
    
        x = tf.concat([state_embedding, rewards], axis=-1)
   
        #
        # pass through backend
        #

        for layer in self.backend_layer_list:
            x = layer(x)

          
        return x

    @tf.function
    def train_step(self, states, rewards, action_target):

        with tf.GradientTape() as tape:
            action_prediction = self(states, rewards)

            
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

        for states, rewards, action_target in dataset:
            action_prediction = self(states, rewards)

            loss = self.loss_function(action_target, action_prediction)
            self.metric_loss.update_state(loss)

            action_prediction = tf.argmax(action_prediction, axis=-1)
            action_target = tf.argmax(action_target, axis=-1)
            self.metric_accuracy.update_state(action_target, action_prediction)