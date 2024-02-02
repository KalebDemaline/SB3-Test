#!/usr/bin/env python
# coding: utf-8
#View Run at: https://wandb.ai/game_wordle/wordle/runs/0j8z4jqq
import random
import wandb
import os
import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.dqn.policies import MlpPolicy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

# Function to select a random word from a text file containing 5-letter words
def choose_word():
    with open("5-letter-words.txt", "r") as file:
        words = file.readlines()
    return random.choice(words).strip()  # Remove leading/trailing whitespace

# Function to check the guessed word against the secret word
def check_word(secret_word, guess):
    return secret_word == guess

# Function to provide feedback on the guessed word with color coding
def provide_feedback(secret_word, guess):
    feedback = []
    for i in range(len(secret_word)):
        if secret_word[i] == guess[i]:
            feedback.append(f"\033[92m{guess[i]}\033[0m")  # Correct letter in the correct position (green)
        elif guess[i] in secret_word:
            feedback.append(f"\033[93m{guess[i]}\033[0m")  # Correct letter in the wrong position (yellow)
        else:
            feedback.append(f"\033[91m{guess[i]}\033[0m")  # Incorrect letter (red)
    return " ".join(feedback)

class WordleEnv(gym.Env):
    metadata = {'render.modes': ['human', 'console']}

    def __init__(self):
        super(WordleEnv, self).__init__()
        # Load word list from a file
        try:
            with open("5-letter-words.txt", "r") as file:
                self.word_list = [line.strip().lower() for line in file if len(line.strip()) == 5]
            print(f"Loaded {len(self.word_list)} words.")
        except FileNotFoundError:
            print("The file '5-letter-words.txt' was not found.")
            self.word_list = []

        self.action_space = spaces.Discrete(len(self.word_list))
        self.observation_space = spaces.Box(low=0, high=26, shape=(6, 5), dtype=np.int8)

        self.secret_word = None
        self.current_attempt = 0
        self.max_attempts = 6
        self.state = np.zeros((self.max_attempts, 5), dtype=np.int8)
        self.feedback_state = np.zeros((self.max_attempts, 5), dtype=np.int8)
        self.color_feedback = ""

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)

    def reset(self):
        self.secret_word = random.choice(self.word_list)
        self.current_attempt = 0
        self.state = np.zeros((self.max_attempts, 5), dtype=np.int8)
        self.feedback_state = np.zeros((self.max_attempts, 5), dtype=np.int8)
        self.color_feedback = ""
        return self.state

    def step(self, action):
        assert self.action_space.contains(action), "Invalid Action"
        guess = self.word_list[action]
        feedback = self._provide_feedback(self.secret_word, guess)
        guess_encoded = [self._char_to_int(c) for c in guess]

        self.state[self.current_attempt] = guess_encoded
        self.feedback_state[self.current_attempt] = feedback
        self.current_attempt += 1
        done = guess == self.secret_word or self.current_attempt == self.max_attempts
        reward = self._calculate_reward(guess, done)

        self.color_feedback = self.provide_feedback(self.secret_word, guess)

        return self.state, reward, done, {}

    def render(self, mode='human'):
        if mode == 'console':
            print(self.color_feedback)
        else:
            pass

    def _provide_feedback(self, secret_word, guess):
        feedback = np.zeros(5, dtype=np.int8)
        for i in range(len(secret_word)):
            if secret_word[i] == guess[i]:
                feedback[i] = 2
            elif guess[i] in secret_word:
                feedback[i] = 1
        return feedback

    def _calculate_reward(self, guess, done):
        reward = 0
        if guess == self.secret_word:
            reward = 20
        else:
            for i in range(len(guess)):
                if guess[i] == self.secret_word[i]:
                    reward += 1
                elif guess[i] in self.secret_word:
                    reward += 0.5
            if done:
                reward -= 2
        return reward

    def _char_to_int(self, char):
        return ord(char) - ord('a') + 1

    def provide_feedback(self, secret_word, guess):
        feedback = []
        for i in range(len(secret_word)):
            if secret_word[i] == guess[i]:
                feedback.append(f"\033[92m{guess[i]}\033[0m")  # Green
            elif guess[i] in secret_word:
                feedback.append(f"\033[93m{guess[i]}\033[0m")  # Yellow
            else:
                feedback.append(f"\033[91m{guess[i]}\033[0m")  # Red
        return " ".join(feedback)

if __name__ == "__main__":
    env = WordleEnv()
    obs = env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render(mode='console')
        if done:
            obs = env.reset()


# Create the environment
env = WordleEnv()
vec_env = make_vec_env(lambda: env, n_envs=4)

import os

hostname = os.uname()[1]

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="wordle",
    name = hostname + ":" + wandb.util.generate_id(),

    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.0001,
    "architecture": "DQN",
    "monitor_gym": True
    }
)

# simulate training
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset
    
    # log metrics to wandb
    wandb.log({"acc": acc, "loss": loss})



# Define the checkpoint and evaluation callbacks
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./models/', name_prefix='wordle_model')
eval_callback = EvalCallback(vec_env, best_model_save_path='./models/', log_path='./logs/', eval_freq=500, deterministic=True, render=False)

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo.policies import MlpPolicy

# Create the environment
env = WordleEnv()
vec_env = make_vec_env(lambda: env, n_envs=4)


model = PPO(MlpPolicy, vec_env, learning_rate=5e-5, 
            n_steps=1024, batch_size=64, n_epochs=15, gamma=0.95, 
            gae_lambda=0.9, clip_range=0.2, clip_range_vf=None,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
            use_sde=False, sde_sample_freq=-1, target_kl=None,
            tensorboard_log=None, policy_kwargs=None, verbose=1, 
            seed=None, device='auto', _init_setup_model=True)


# Train the agent with a reduced number of timesteps
model.learn(total_timesteps=300000)

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=10)

# Print the evaluation result
print(f"Mean reward: {mean_reward}, Std Reward: {std_reward}")

# Save the model (optional)
model.save("wordle_sample")

# Close the environment
env.close()

# Load the trained agent 
model = PPO.load("wordle_sample")

# Evaluation
num_episodes = 1000
success_count = 0
total_reward = 0

for episode in range(num_episodes):
    obs = env.reset()
    done = False
    episode_reward = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        if done and reward > 0:  # Successful guess
            success_count += 1
    total_reward += episode_reward

success_rate = success_count / num_episodes
average_reward = total_reward / num_episodes

print(f"Success Rate: {success_rate * 100:.2f}%")
print(f"Average Reward: {average_reward:.2f}")

# Finish the W&B run
wandb.finish()
