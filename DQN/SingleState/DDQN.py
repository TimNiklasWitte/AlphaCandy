import tensorflow as tf


class DDDQN(tf.keras.Model):
    def __init__(self, num_actions: int):
        """Init the DDDQN. 

        Keyword arguments:
        num_actions -- Number of possible actions which can be taken in the gym.
        """
        super(DDDQN, self).__init__()

  
        self.layer_list = [
            tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=(2,2), activation="tanh", padding='valid'),
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), activation="tanh", padding='valid'),
            tf.keras.layers.Flatten(),
        ]

        self.v = tf.keras.layers.Dense(1, activation=None, name="state")
        self.a = tf.keras.layers.Dense(num_actions, activation=None, name="adventage")

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.loss_function = tf.keras.losses.MeanSquaredError()

    @tf.function
    def call(self, x: tf.Tensor):
        """Forward pass through the network. 

        Keyword arguments:
        x -- network input

        Return:
        network output
        """

  
        for layer in self.layer_list:
            x = layer(x)

        v = self.v(x)
        a = self.a(x)
        q = v + (a - tf.math.reduce_mean(a, axis=1, keepdims=True))


        return q
            

    @tf.function
    def train_step(self, x: tf.Tensor, target: tf.Tensor):
        """Train the network based on input and target,

        Keyword arguments:
        x -- network input
        target -- target
        """
        
        with tf.GradientTape() as tape:
            predictions = self(x, training=True)
            loss = self.loss_function(target, predictions)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
