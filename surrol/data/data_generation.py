"""
Data generation for the case of Psm Envs and demonstrations.
Refer to
https://github.com/openai/baselines/blob/master/baselines/her/experiment/data_generation/fetch_data_generation.py
"""
import os
import argparse
import gym
import time
import numpy as np
import imageio
from surrol.const import ROOT_DIR_PATH
from datetime import datetime  # 1. Import datetime for timestamps

parser = argparse.ArgumentParser(description='generate demonstrations for imitation')
parser.add_argument('--env', type=str, required=True,
                    help='the environment to generate demonstrations')
parser.add_argument('--video', action='store_true',
                    help='whether or not to record video')
parser.add_argument('--steps', type=int,
                    help='how many steps allowed to run')
args = parser.parse_args()

actions = []
observations = []
infos = []

images = []  # record video
masks = []

def main():
    env = gym.make(args.env, render_mode='human')  # 'human'
    num_itr = 100 if not args.video else 1
    cnt = 0
    init_state_space = 'random'
    env.reset()
    print("Reset!")
    init_time = time.time()

    if args.steps is None:
        args.steps = env._max_episode_steps

    # 2. Initialize counters
    total_trials = 0
    failed_epochs = 0

    print()
    while len(actions) < num_itr:
        obs = env.reset()
        print("ITERATION NUMBER ", len(actions))
        success = goToGoal(env, obs)
        total_trials += 1  # Increment total trials
        if not success:
            failed_epochs += 1  # Increment failed epochs
        cnt += 1

    file_name = "data_"
    file_name += args.env
    file_name += "_" + init_state_space
    file_name += "_" + str(num_itr)
    file_name += ".npz"

    folder = 'demo' if not args.video else 'video'
    folder = os.path.join(ROOT_DIR_PATH, 'data', folder)

    np.savez_compressed(os.path.join(folder, file_name),
                        acs=actions, obs=observations, info=infos)  # save the file

    if args.video:
        video_name = "video_"
        video_name += args.env + ".mp4"
        writer = imageio.get_writer(os.path.join(folder, video_name), fps=20)
        for img in images:
            writer.append_data(img)
        writer.close()

        if len(masks) > 0:
            mask_name = "mask_"
            mask_name += args.env + ".npz"
            np.savez_compressed(os.path.join(folder, mask_name),
                                masks=masks)  # save the file

    used_time = time.time() - init_time
    print("Saved data at:", folder)
    print("Time used: {:.1f}m, {:.1f}s\n".format(used_time // 60, used_time % 60))
    print(f"Trials: {num_itr}/{cnt}")

    # 3. Log the results
    log_filepath = os.path.join(ROOT_DIR_PATH, 'data', 'data_generation.log')  # Define log file path
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Current timestamp

    with open(log_filepath, 'a') as log_file:
        log_file.write(f"Timestamp: {timestamp}\n")
        log_file.write(f"Environment: {args.env}\n")
        log_file.write(f"Total Trials Attempted: {total_trials}\n")
        log_file.write(f"Successful Episodes: {len(actions)}\n")
        log_file.write(f"Failed Episodes: {failed_epochs}\n")
        log_file.write(f"Data Saved at: {folder}\n")
        log_file.write(f"Total Time Used: {used_time // 60:.1f}m {used_time % 60:.1f}s\n")
        log_file.write(f"Trials Completed: {cnt}/{num_itr}\n")
        log_file.write("-" * 50 + "\n")  # Separator for readability

    env.close()


def goToGoal(env, last_obs):
    episode_acs = []
    episode_obs = []
    episode_info = []

    time_step = 0  # count the total number of time steps
    episode_init_time = time.time()
    episode_obs.append(last_obs)

    obs, success = last_obs, False

    while time_step < min(env._max_episode_steps, args.steps):
        action = env.get_oracle_action(obs)
        if args.video:
            # img, mask = env.render('img_array')
            img = env.render('rgb_array')
            images.append(img)
            # masks.append(mask)

        obs, reward, done, info = env.step(action)
        # print(f" -> obs: {obs}, reward: {reward}, done: {done}, info: {info}.")
        time_step += 1

        if isinstance(obs, dict) and info.get('is_success', 0) > 0 and not success:
            print("Timesteps to finish:", time_step)
            success = True

        episode_acs.append(action)
        episode_info.append(info)
        episode_obs.append(obs)
    print("Episode time used: {:.2f}s\n".format(time.time() - episode_init_time))

    if success:
        actions.append(episode_acs)
        observations.append(episode_obs)
        infos.append(episode_info)
        return True  # Indicate success
    else:
        return False  # Indicate failure


if __name__ == "__main__":
    main()
