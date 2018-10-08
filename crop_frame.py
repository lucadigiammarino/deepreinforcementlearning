import gym
import tensorflow as tf
import matplotlib.pyplot as plt

# Create a Road Runner environment
env = gym.make('Gopher-v0')
# Reset it, returns the starting frame
frame = env.reset()

# is_done = False
for i in range(100):
    # get the last frame when is game over, after performing random actions
    original_frame, _, is_done, _ = env.step(env.action_space.sample())

frame_tf = tf.Variable(original_frame)
# convert to greyscale - shape=[250,160,3]
output = tf.image.rgb_to_grayscale(frame_tf)
# crop image
output = tf.image.crop_to_bounding_box(output, 110, 0, 120, 160)
# resize image
output = tf.image.resize_images(output, [50, 50], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# removes all dimensions of size 1
output = tf.squeeze(output)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
modified_img = sess.run(output)

# plotting results
fig = plt.figure()
fig.suptitle('Comparing frames', fontsize=14, fontweight='bold')
fig.add_subplot(1, 2, 1)
plt.imshow(modified_img)
fig.add_subplot(1, 2, 2)
plt.imshow(original_frame)
plt.show()