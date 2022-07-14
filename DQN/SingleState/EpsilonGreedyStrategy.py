class EpsilonGreedyStrategy:
    def __init__(self, start: float, end: float, decay: float):
        """
        Init EpsilonGreedyStrategy. Epsilon first value is start. 
        Calling reduce_epsilon will reduce Epsilon (multiplicative decay).
        Its min value will be end.

        Keyword arguments:
            start -- start value of Epsilon
            end -- min value of Epsilon
            decay -- multiplicative decay term
        """

        self.epsilon = start
        self.end = end

        self.decay = decay

    def get_exploration_rate(self):
        """
        Return the current Epsilon value
        """
        return self.epsilon

    def reduce_epsilon(self):
        
        """
        Multiplicative decay of Epsilon by a factor definied in decay parameter (see constructor).
        Note after Epsilon reached the min value (see end parameter in constructor), there will 
        be no decay.
        """

        new_epsilon = self.epsilon * self.decay
        if self.end < new_epsilon:
            self.epsilon = new_epsilon

       
