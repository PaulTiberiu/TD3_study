#!/bin/bash

reward_files=()

# Loop to run TD3 for different seeds
for seed in {1..10}
do
    # Run the TD3 algorithm
    python cleanrl/td3_continuous_action.py \
        --seed $seed \
        --env-id LunarLanderContinuous-v2 \
        --total-timesteps 100000 \
        --learning-rate 0.001 \
        --buffer-size 200000 \
        --learning-starts 10000 \
        --gamma 0.98 \
        --batch-size 64 \
        --policy-noise 0.1 \
        --exploration-noise 0.1 \
        --tau 0.005 \
        --noise-clip 0.5 \
        --save-model  # Automatically saves the model in the runs/{run_name} folder
    reward_file="runs/LunarLanderContinuous-v2__td3_continuous_action__${seed}__*/episode_rewards.npy"
    reward_files+=($reward_file)
done

# Combine and average the rewards across runs using Python
python -c "
import numpy as np
import matplotlib.pyplot as plt

# Load all reward files into a list of lists
reward_files = '''${reward_files[@]}'''.split()
all_rewards = [np.load(f) for f in reward_files]

# Stack rewards along axis 0 (episodes) to calculate mean rewards across seeds
rewards_array = np.vstack(all_rewards)

# Calculate mean rewards over the seeds
mean_rewards = np.mean(rewards_array, axis=0)

# Plot the mean learning curve
plt.plot(mean_rewards)
plt.xlabel('Episodes')
plt.ylabel('Mean Reward')
plt.title('TD3 Learning Curve using CleanRL over 10 seeds')
plt.grid(True)
plt.show()
"