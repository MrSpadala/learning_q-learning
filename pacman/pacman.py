import tensorflow as tf
import numpy as np
import gym
import sys
import os
from time import sleep


if len(sys.argv) != 2 or (sys.argv[1]!='train' and sys.argv[1]!='play'):
    print('ARGUMENTS: "train" to train network, "play" to play games')
    exit()



env = gym.make("MsPacman-v0")
obs = env.reset()

input_height = 88
input_width = 80
input_channels = 1
conv_n_maps = [32, 64, 64]
conv_kernel_sizes = [(8,8), (4,4), (3,3)]
conv_strides = [4, 2, 1]
conv_paddings = ["SAME"] * 3 
conv_activation = [tf.nn.relu] * 3
n_hidden_in = 64 * 11 * 10  # conv3 has 64 maps of 11x10 each
n_hidden = 512
hidden_activation = tf.nn.relu
n_outputs = env.action_space.n  # 9 discrete actions are available
initializer = tf.contrib.layers.variance_scaling_initializer()



mspacman_color = 210 + 164 + 74

def preprocess_observation(obs):
    img = obs[1:176:2, ::2] # crop and downsize
    img = img.sum(axis=2) # to greyscale
    img[img==mspacman_color] = 0 # Improve contrast
    img = (img // 3 - 128).astype(np.int8) # normalize from -128 to 127
    return img.reshape(88, 80, 1)

img = preprocess_observation(obs)




def q_network(X_state, name):
    prev_layer = X_state / 128.0 # scale pixel intensities to the [-1.0, 1.0] range.
    with tf.variable_scope(name) as scope:
        for n_maps, kernel_size, strides, padding, activation in zip(
                conv_n_maps, conv_kernel_sizes, conv_strides,
                conv_paddings, conv_activation):
            prev_layer = tf.layers.conv2d(
                prev_layer, filters=n_maps, kernel_size=kernel_size,
                strides=strides, padding=padding, activation=activation,
                kernel_initializer=initializer)
        last_conv_layer_flat = tf.reshape(prev_layer, shape=[-1, n_hidden_in])
        hidden = tf.layers.dense(last_conv_layer_flat, n_hidden,
                                 activation=hidden_activation,
                                 kernel_initializer=initializer)
        outputs = tf.layers.dense(hidden, n_outputs,
                                  kernel_initializer=initializer)
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope=scope.name)
    trainable_vars_by_name = {var.name[len(scope.name):]: var
                              for var in trainable_vars}
    return outputs, trainable_vars_by_name


X_state = tf.placeholder(tf.float32, shape=[None, input_height, input_width,
                                            input_channels])
online_q_values, online_vars = q_network(X_state, name="q_networks/online")
target_q_values, target_vars = q_network(X_state, name="q_networks/target")

copy_ops = [target_var.assign(online_vars[var_name])
            for var_name, target_var in target_vars.items()]
copy_online_to_target = tf.group(*copy_ops)



learning_rate = 0.001
momentum = 0.95

with tf.variable_scope("train"):
    X_action = tf.placeholder(tf.int32, shape=[None])
    y = tf.placeholder(tf.float32, shape=[None, 1])
    q_value = tf.reduce_sum(online_q_values * tf.one_hot(X_action, n_outputs),
                            axis=1, keep_dims=True)
    error = tf.abs(y - q_value)
    clipped_error = tf.clip_by_value(error, 0.0, 1.0)
    linear_error = 2 * (error - clipped_error)
    loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)

    global_step = tf.Variable(0, trainable=False, name='global_step')
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True)
    training_op = optimizer.minimize(loss, global_step=global_step)

init = tf.global_variables_initializer()
saver = tf.train.Saver()



from collections import deque

replay_memory_size = 500000
replay_memory = deque([], maxlen=replay_memory_size)

def sample_memories(batch_size):
    indices = np.random.permutation(len(replay_memory))[:batch_size]
    cols = [[], [], [], [], []] # state, action, reward, next_state, continue
    for idx in indices:
        memory = replay_memory[idx]
        for col, value in zip(cols, memory):
            col.append(value)
    cols = [np.array(col) for col in cols]
    return cols[0], cols[1], cols[2].reshape(-1, 1), cols[3], cols[4].reshape(-1, 1)




eps_min = 0.1
eps_max = 1.0
eps_decay_steps = 2000000

def epsilon_greedy(q_values, step):
    epsilon = max(eps_min, eps_max - (eps_max-eps_min) * step/eps_decay_steps)
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs) # random action
    else:
        return np.argmax(q_values) # optimal action





n_steps = 4000000  # total number of training steps
training_start = 10000  # start training after 10,000 game iterations
training_interval = 4  # run a training step every 4 game iterations
save_steps = 1000  # save the model every 1,000 training steps
copy_steps = 10000  # copy online DQN to target DQN every 10,000 training steps
discount_rate = 0.95
skip_start = 90  # Skip the start of every game (it's just waiting time).
batch_size = 50
iteration = 0  # game iterations
checkpoint_path = "./my_dqn.ckpt"
done = True # env needs to be reset


loss_val = np.infty
game_length = 0
total_max_q = 0
mean_max_q = 0.0



# Pacman environment revisited
lives = 3  #it gets resetted immediately when game is done
reward_game_over = -500   #game over penalty
reward_game_win  = 500    #game won reward
reward_max = 10           #max reward for a non-winning action
skip_game_over_frames = 15

if sys.argv[1]=='train':
    #Training

    with tf.Session() as sess:
        if os.path.isfile(checkpoint_path + ".index"):
            saver.restore(sess, checkpoint_path)
        else:
            init.run()
            copy_online_to_target.run()
        while True:
            step = global_step.eval()
            if step > n_steps:
                break
            iteration += 1
            print("\rIteration {}\tTraining step {}/{} ({:.1f})%\tLoss {:5f}\tMean Max-Q {:5f}   ".format(
                iteration, step, n_steps, step * 100 / n_steps, loss_val, mean_max_q), end="")
            if done: # game over, start again
                obs = env.reset()
                lives = env.step(0)[3]['ale.lives']   # starting lives = info['ale.lives']
                for skip in range(skip_start-1): # skip the start of each game
                    obs, reward, done, info = env.step(0)
                state = preprocess_observation(obs)

            # Online DQN evaluates what to do
            q_values = online_q_values.eval(feed_dict={X_state: [state]})
            action = epsilon_greedy(q_values, step)

            # Online DQN plays
            obs, reward, done, info = env.step(action)
            next_state = preprocess_observation(obs)
            # Changed rewards
            if reward > 10:     # eating a ghost gives too much points
                reward = reward_max
            if info['ale.lives'] < lives:
                for _ in range(skip_game_over_frames):
                    replay_memory.pop()
                reward = reward_game_over
                state = replay_memory[-1][0]
                action = replay_memory[-1][1]
                lives = info['ale.lives']
            if done and info['ale.lives'] > 0:
                reward = reward_game_win

            # Let's memorize what happened
            replay_memory.append((state, action, reward, next_state, 1.0 - done))
            state = next_state

            # Compute statistics for tracking progress (not shown in the book)
            total_max_q += q_values.max()
            game_length += 1
            if done:
                mean_max_q = total_max_q / game_length
                total_max_q = 0.0
                game_length = 0

            if iteration < training_start or iteration % training_interval != 0:
                continue # only train after warmup period and at regular intervals
            
            # Sample memories and use the target DQN to produce the target Q-Value
            X_state_val, X_action_val, rewards, X_next_state_val, continues = (
                sample_memories(batch_size))
            next_q_values = target_q_values.eval(
                feed_dict={X_state: X_next_state_val})
            max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
            y_val = rewards + continues * discount_rate * max_next_q_values

            # Train the online DQN
            _, loss_val = sess.run([training_op, loss], feed_dict={
                X_state: X_state_val, X_action: X_action_val, y: y_val})

            # Regularly copy the online DQN to the target DQN
            if step % copy_steps == 0:
                copy_online_to_target.run()

            # And save regularly
            if step % save_steps == 0:
                saver.save(sess, checkpoint_path)
                f = open('stats', 'a')
                f.write(str(step)+' '+str(mean_max_q)+'\n')
                f.close()
                f = open('loss', 'a')
                f.write(str(step)+' '+str(loss_val)+'\n')
                f.close()


elif sys.argv[1]=='play':
    n_games = 100
    def play():
        with tf.Session() as sess:
            saver.restore(sess, checkpoint_path)
            obs = env.reset()
            score = 0
            while True:
                q_values = online_q_values.eval(feed_dict={X_state: [preprocess_observation(obs)]})
                action = np.argmax(q_values)
                obs, reward, done, info = env.step(action)
                env.render()  #rendering slows down
                if done:
                    break
                score += reward
                #print(str(max(q_values))+' '+str(reward))
        return score
    s = []
    for _ in range(5):
        s.append(play())

    print('Average on ',n_games,' games;', np.average(s))

else:
    print('invalid argument '+sys.argv[1])
    exit()



