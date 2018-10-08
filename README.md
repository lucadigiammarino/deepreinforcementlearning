# Deep Reinforcement Learning for Atari game

The problem is based on allowing the agent to learn a *policy* using CNN and RL techniques. The environment taken into consideration is an Atari game called **Gopher** provided by the gym package (Open AI, 2017). This game has a discrete action space and considerably resizable observation space.

The problem can be divided into two subsections:
1.	Working with gym to model Atari environments
1.	Use a CNN to learn Q-function

### Working with gym to model Atari environments

Gym is a framework for developing and comparing reinforcement learning algorithms. It makes no assumption about the structure of the agent and has compatibility with most of the numerical computation libraries like Tensorflow. It provides Atari games by integrating ALE (Arcade Learning Environment) in a straightforward form.

In order for the agent to perform specific actions the step function provided by gym is fundamental. This function returns four values:
1.	**observation** (type : environment specific object) representing the observation of the environment.
1.	**reward** (type: float) representing the amount of reward achieved by the previous action.
1.	**done** (type: boolean) indicates when the episode is terminated because the agent lost the game.
1.	**info** (type: dict) diagnostic information useful only for debugging purposes.

Videos have been recorded at each 50 training steps to check the progress of learning.

### Use a CNN to learn Q-function

A convolutional neural network is used as a non-linear function approximator, being able to approximate function with large state space such as the number of pixels in the game screen. The CNN has been designed following the architecture and the parameters released by [Deep Mind](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf). 

However, reinforcement learning presents several challenges from a deep learning perspective. RL algorithms needs to learn from a scalar policy that is frequently noisy and delayed, conversely to classification problems with labelled data. In addition, in RL there are not independent data samples but sequences of correlated states. Moreover, in RL the data distribution changes as the algorithm learns new behaviours, on the other hand, deep learning standard methods assumes a fixed distribution. 

According to Deep Mind a Convolutional Neural Network can overcome these challenges to learn successful control policies from raw video data in complex RL arcade environments. In brief, the network is trained with a variant of the Q-learning algorithm using stochastic gradient descent to update the weights. To reduce the problem of sequential states and dynamic distributions it is important to use an experience replay mechanism which smooths training distributions over many past behaviours by sampling previous transitions. 

If you are not familiar with these basic concepts of Reinforcement Learning check the [Sutton and Burton Book - free pdf](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf).

## Preprocessing

Input frames have been pre-processed before since working with raw Atari images could be computational demanding (250 * 160). Therefore, the colour input has first been converted to grayscale. After, the image has been cropped since most of the upper background was useless for the agent to maximise his reward (this reduced the image to 120 * 160). Finally, the frame has been reduced further to a square input of 50 * 50 according to the CNN’s implementation. As mentioned previously the last 4 frames are stack together as a single input, hence the volume is composed by 50 * 50 * 4. The implemented ```crop_frame.py``` file allowed to perfectly remove the unnecessary and test a right resolution of the input image [Preprocessing Gopher Atari Input] ().




