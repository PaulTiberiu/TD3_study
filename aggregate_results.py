import os
import re
import numpy as np
import tensorflow as tf
from tensorboard.backend.event_processing import event_accumulator

def load_metrics(logdir):
    event_acc = event_accumulator.EventAccumulator(logdir)
    event_acc.Reload()  # Loads all events

    if 'charts/episodic_return' in event_acc.Tags()['scalars']:
        rewards = event_acc.Scalars('charts/episodic_return')
        print("djksabdisabidgbhsabdabsidbiasbdbsajsdasidbsia")
        print(len(rewards))
        # Collect all reward values
        reward_values = np.array([reward.value for reward in rewards])
        
        return reward_values
    else:
        print(f"No 'charts/episodic_return' found in {logdir}")
        return np.array([])

# Directory where your runs are saved
base_dir = 'runs/'

# List to hold all rewards
all_rewards = []

# Regex pattern to match the folder structure
pattern = r'^(LunarLanderContinuous-v2)__td3_continuous_action__(?P<seed>\d+)__\d+$'

# Loop through the directories in the base_dir
for folder in os.listdir(base_dir):
    match = re.match(pattern, folder)
    if match:
        logdir = os.path.join(base_dir, folder)
        if os.path.exists(logdir):
            rewards = load_metrics(logdir)
            print(f"Found {len(rewards)} rewards in {logdir}")
            if rewards.size > 0:
                all_rewards.append(rewards)
        else:
            print(f"Log directory {logdir} does not exist.")

# If there are any valid runs
if len(all_rewards) > 0:
    # Find the minimum length of the reward arrays
    min_length = min([len(reward) for reward in all_rewards])

    # Trim all reward arrays to the minimum length
    trimmed_rewards = [reward[:min_length] for reward in all_rewards]

    # Convert to numpy array
    all_rewards = np.array(trimmed_rewards)

    # Calculate the mean and std across runs
    mean_rewards = np.mean(all_rewards, axis=0)
    std_rewards = np.std(all_rewards, axis=0)

    # Create a new directory for mean results
    mean_logdir = os.path.join(base_dir, 'mean_results')
    os.makedirs(mean_logdir, exist_ok=True)

    # Log the mean rewards to TensorBoard
    with tf.summary.create_file_writer(os.path.join(mean_logdir, 'mean_rewards')).as_default():
        for i, (mean, std) in enumerate(zip(mean_rewards, std_rewards)):
            tf.summary.scalar('mean_reward', mean, step=i)
            tf.summary.scalar('std_reward', std, step=i)

    print("Mean rewards logged to:", mean_logdir)
else:
    print("No valid rewards found across runs.")

# import numpy as np
# import os
# from tensorboard.backend.event_processing import event_accumulator

# # Fonction pour charger les métriques à partir des logs TensorBoard
# def load_metrics(logdir):
#     event_acc = event_accumulator.EventAccumulator(logdir)
#     event_acc.Reload()  # Charge tous les événements

#     print(f"Chargement des métriques depuis : {logdir}")
#     if 'charts/episodic_return' in event_acc.Tags()['scalars']:
#         rewards = event_acc.Scalars('charts/episodic_return')
#         # Collecter toutes les valeurs de récompenses
#         reward_values = np.array([reward.value for reward in rewards])
#         return reward_values
#     else:
#         print(f"Aucune valeur 'charts/episodic_return' trouvée dans {logdir}")
#         return np.array([])

# # Fonction principale pour récupérer les récompenses de tous les répertoires et calculer la moyenne
# def aggregate_rewards(base_dir):
#     all_rewards = []

#     # Parcourir tous les sous-répertoires dans le répertoire de base
#     for dir_name in os.listdir(base_dir):
#         full_path = os.path.join(base_dir, dir_name)
#         if os.path.isdir(full_path):  # Vérifie si c'est un répertoire
#             rewards = load_metrics(full_path)
#             if rewards.size > 0:
#                 all_rewards.append(rewards)

#     # Si des récompenses ont été collectées, calculez la moyenne
#     if all_rewards:
#         max_length = max(map(len, all_rewards))  # Longueur de la série de récompenses la plus longue
#         # Remplir les tableaux de récompenses avec NaN pour les longueurs différentes
#         padded_rewards = [np.pad(reward, (0, max_length - len(reward)), constant_values=np.nan) for reward in all_rewards]
        
#         mean_rewards = np.nanmean(padded_rewards, axis=0)
#         return mean_rewards
#     else:
#         print("Aucune récompense trouvée dans les répertoires.")
#         return np.array([])

# # Exemple d'utilisation
# base_dir = 'runs'  # Remplacez par le chemin vers votre répertoire
# mean_rewards = aggregate_rewards(base_dir)

# # Afficher les résultats
# print("Moyenne des récompenses :", mean_rewards)


# import numpy as np
# import os
# import matplotlib.pyplot as plt
# from tensorboard.backend.event_processing import event_accumulator

# # Fonction pour charger les métriques à partir des logs TensorBoard
# def load_metrics(logdir):
#     event_acc = event_accumulator.EventAccumulator(logdir)
#     event_acc.Reload()  # Charge tous les événements

#     print(f"Chargement des métriques depuis : {logdir}")
#     if 'charts/episodic_return' in event_acc.Tags()['scalars']:
#         rewards = event_acc.Scalars('charts/episodic_return')
#         # Collecter toutes les valeurs de récompenses
#         reward_values = np.array([reward.value for reward in rewards])
#         return reward_values
#     else:
#         print(f"Aucune valeur 'charts/episodic_return' trouvée dans {logdir}")
#         return np.array([])

# # Fonction principale pour récupérer les récompenses de tous les répertoires et calculer la moyenne
# def aggregate_rewards(base_dir):
#     all_rewards = []

#     # Parcourir tous les sous-répertoires dans le répertoire de base
#     for dir_name in os.listdir(base_dir):
#         full_path = os.path.join(base_dir, dir_name)
#         if os.path.isdir(full_path):  # Vérifie si c'est un répertoire
#             rewards = load_metrics(full_path)
#             if rewards.size > 0:
#                 all_rewards.append(rewards)

#     # Si des récompenses ont été collectées, calculez la moyenne
#     if all_rewards:
#         max_length = max(map(len, all_rewards))  # Longueur de la série de récompenses la plus longue
#         # Remplir les tableaux de récompenses avec NaN pour les longueurs différentes
#         padded_rewards = [np.pad(reward, (0, max_length - len(reward)), constant_values=np.nan) for reward in all_rewards]
        
#         mean_rewards = np.nanmean(padded_rewards, axis=0)
#         print(mean_rewards)
#         return mean_rewards
#     else:
#         print("Aucune récompense trouvée dans les répertoires.")
#         return np.array([])

# # Exemple d'utilisation
# base_dir = 'runs'  # Remplacez par le chemin vers votre répertoire
# mean_rewards = aggregate_rewards(base_dir)

# # Si la moyenne des récompenses est disponible, nous l'affichons
# if mean_rewards.size > 0:
#     total_timesteps = 100000  # Nombre total de timesteps
#     num_episodes = len(mean_rewards)  # Nombre d'épisodes récupérés
#     timesteps_per_episode = total_timesteps / num_episodes  # Timesteps par épisode

#     # Créer un axe des x (timesteps) basé sur le nombre d'épisodes
#     x_axis = np.arange(0, total_timesteps, timesteps_per_episode)

#     # Tracer les récompenses moyennes avec Matplotlib
#     plt.plot(x_axis, mean_rewards)
#     plt.xlabel('Timesteps')
#     plt.ylabel('Mean Reward')
#     plt.title('Mean Reward vs Timesteps over 100,000 Timesteps')
#     plt.grid(True)
#     plt.show()
# else:
#     print("Pas de données de récompense moyennes à afficher.")
