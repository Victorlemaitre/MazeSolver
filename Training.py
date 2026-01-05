from MazeGenerator_cython import grid_to_img
from MazeWorld_cython import MazeWorld_cython as MazeWorld
from PPO import PPO_agent
import numpy as np
import torch
from tqdm import tqdm
import imageio
import argparse

parser = argparse.ArgumentParser(description="PPO training on MazeWorld")

# --- General / IO ---
parser.add_argument("--model-name", type=str, default="model.pth")

# --- Environment ---
parser.add_argument("--maze-size", type=int, default=7)
parser.add_argument("--num-envs", type=int, default=10)
parser.add_argument("--max-steps-per-episode", type=int, default=64)
parser.add_argument("--nb-channels", type=int, default=4)
parser.add_argument("--init-nb-end-steps", type=int, default=2)

# --- Training loop ---
parser.add_argument("--n-updates", type=int, default=5000)
parser.add_argument("--print-every", type=int, default=50)
parser.add_argument("--nb-batches", type=int, default=4)

# --- PPO hyperparameters ---
parser.add_argument("--gamma", type=float, default=0.995)
parser.add_argument("--gae-lambda", type=float, default=0.95)
parser.add_argument("--learning-rate", type=float, default=1e-3)
parser.add_argument("--entropy-coeff", type=float, default=0.01)
parser.add_argument("--clip-eps", type=float, default=0.2)
parser.add_argument("--target-kl", type=float, default=0.02)
parser.add_argument("--n-passes", type=int, default=3)
parser.add_argument("--critic-coeff", type=float, default=1.0)

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tqdm.write(f"Using : {device}")

if __name__ == '__main__':

    MODEL_NAME = args.model_name

    MAZE_SIZE = args.maze_size
    NUM_ENVS = args.num_envs
    MAX_STEPS_PER_EPISODE = args.max_steps_per_episode
    STEPS_PER_UPDATE = MAX_STEPS_PER_EPISODE
    NB_CHANNELS = args.nb_channels

    GAMMA = args.gamma
    GAE_LAMBDA = args.gae_lambda
    LEARNING_RATE = args.learning_rate
    ENTROPY_COEFF = args.entropy_coeff
    CLIP_EPS = args.clip_eps
    TARGET_KL = args.target_kl
    N_PASSES = args.n_passes
    CRITIC_COEFF = args.critic_coeff

    NB_BATCHES = args.nb_batches
    BATCH_SIZE = (STEPS_PER_UPDATE * NUM_ENVS) // NB_BATCHES

    N_UPDATES = args.n_updates
    PRINT_EVERY_N_UPDATES = args.print_every
    NB_END_STEPS = args.init_nb_end_steps

    BASE_MAX_STEPS = MAX_STEPS_PER_EPISODE

    # =========================
    # Rollout buffers
    # =========================

    ep_terminated = np.empty((STEPS_PER_UPDATE, NUM_ENVS), dtype=np.uint8)
    ep_truncated = np.empty((STEPS_PER_UPDATE, NUM_ENVS), dtype=np.uint8)
    ep_rewards_np = np.empty((STEPS_PER_UPDATE, NUM_ENVS), dtype=np.float32)

    # =========================
    # Environment
    # =========================

    envs = MazeWorld(
        num_envs=NUM_ENVS,
        maze_size=MAZE_SIZE,
        init_nb_step_end=NB_END_STEPS,
        max_step=MAX_STEPS_PER_EPISODE,
        n_steps_per_update=STEPS_PER_UPDATE,
        terminated=ep_terminated,
        truncated=ep_truncated,
        reward=ep_rewards_np,
    )

    # =========================
    # PPO Agent
    # =========================

    agent = PPO_agent(
        maze_size=MAZE_SIZE,
        nb_channel=NB_CHANNELS,
        device=device,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        entropy_coeff=ENTROPY_COEFF,
        eps=CLIP_EPS,
        target_KL=TARGET_KL,
        n_pass=N_PASSES,
        critic_coeff=CRITIC_COEFF,
    )

    agent = torch.compile(agent)

    # Monitoring variables
    avrg_reward_list = []
    ep_length = []
    ep_length_show = []
    last_avrg_step_length_show = MAX_STEPS_PER_EPISODE + 1
    difficulty_increased = False
    idx_print = 0

    # Rollout storage
    ep_done = np.zeros((STEPS_PER_UPDATE, NUM_ENVS), dtype=bool)

    ep_obs = torch.zeros(
        STEPS_PER_UPDATE, NUM_ENVS, NB_CHANNELS, MAZE_SIZE, MAZE_SIZE,
        dtype=torch.float32, device=device
    )
    ep_actions = torch.zeros(STEPS_PER_UPDATE, NUM_ENVS, device=device, dtype=torch.long)
    ep_log_probs = torch.zeros(STEPS_PER_UPDATE, NUM_ENVS, device=device)
    ep_values = torch.zeros(STEPS_PER_UPDATE + 1, NUM_ENVS, device=device)

    tqdm.write("Starting training")
    for sample_phase in tqdm(range(1, N_UPDATES + 1)):
        idx_print += 1

        if sample_phase == 1:
            envs.reset()

        for step in range(STEPS_PER_UPDATE):
            ep_obs[step] = torch.as_tensor(envs._grid_arr, dtype=torch.float32, device=device)

            with torch.no_grad():
                actions, values, log_probs = agent.get_action_value(ep_obs[step])

            envs.step(actions.cpu().numpy().astype(np.int32))

            ep_actions[step] = actions
            ep_log_probs[step] = log_probs
            ep_values[step] = values

            ep_done[step] = np.logical_or(ep_terminated[step], ep_truncated[step])

            if np.any(ep_done[step]):
                step_count = envs._step_count_arr[ep_done[step]].tolist()
                ep_length.extend(step_count)
                ep_length_show.extend(step_count)

        with torch.no_grad():
            last_obs = torch.as_tensor(envs._grid_arr, dtype=torch.float32, device=device)
            last_values = agent.get_value(last_obs)
            ep_values[STEPS_PER_UPDATE] = last_values

        masks_done = torch.from_numpy(~ep_done).to(device)

        ####################################
        # GAE computation
        ####################################
        advantage = np.zeros((STEPS_PER_UPDATE, NUM_ENVS), dtype=np.float32)
        ep_values_np = ep_values.detach().cpu().numpy()

        gae = 0.0
        td_error = (
            ep_rewards_np
            + GAMMA * ep_values_np[1:] * (1 - ep_terminated)
            - ep_values_np[:-1]
        )

        for t in reversed(range(len(ep_rewards_np))):
            gae = td_error[t] + GAMMA * GAE_LAMBDA * gae * (1 - ep_done[t])
            advantage[t] = gae

        advantage = torch.from_numpy(advantage).to(device)

        agent.update(ep_obs, ep_actions, ep_values, ep_log_probs, advantage, masks_done)

        if len(ep_length) > min(30, NUM_ENVS):
            avrg_ep_length = sum(ep_length[-(NUM_ENVS + 30):]) / len(ep_length[-(NUM_ENVS + 30):])

            if avrg_ep_length <= 2 * NB_END_STEPS + NB_END_STEPS // 12 + 2:
                NB_END_STEPS += 1
                envs.set_difficulty(NB_END_STEPS)
                envs.max_step = BASE_MAX_STEPS + 4 * (NB_END_STEPS - 2)
                envs.reset()

                ep_length = []
                ep_length_show = []
                avrg_reward_list = []
                difficulty_increased = True

                tqdm.write(
                    f"|||||||||||||||| difficulty increased to : {NB_END_STEPS} : "
                    f"previous avrg ep length {avrg_ep_length:.2f} ||||||||||||||||||||"
                )

                if PRINT_EVERY_N_UPDATES - idx_print < 10:
                    idx_print = PRINT_EVERY_N_UPDATES - 10

        avrg_reward_list.append(ep_rewards_np.mean())

        if idx_print == PRINT_EVERY_N_UPDATES:
            avrg_reward = sum(avrg_reward_list) / len(avrg_reward_list) if avrg_reward_list else 0.0
            avrg_ep_length_show = sum(ep_length_show) / len(ep_length_show) if ep_length_show else float("nan")

            tqdm.write(f"current distance between start and end cell : {NB_END_STEPS}")
            tqdm.write(f"avrg reward in the last {PRINT_EVERY_N_UPDATES} steps is {avrg_reward}")
            tqdm.write(f"avrg length of episode {avrg_ep_length_show}")
            tqdm.write("================================================")

            if (not difficulty_increased) and (avrg_ep_length_show > (last_avrg_step_length_show - 1e-3)):
                agent.scheduler.step()
                tqdm.write(
                    f"step the scheduler, current learning rate is : "
                    f"{agent.scheduler.get_last_lr()[0]}"
                )
                tqdm.write("-------------------------------------------------")

            last_avrg_step_length_show = avrg_ep_length_show
            ep_length_show = []
            difficulty_increased = False
            idx_print = 0
        # we do not care about being precise here, we just need a realistic upped bound on the maximum length of a path in our maze
        if NB_END_STEPS >= ((MAZE_SIZE - 2) // 2 + 1) * (MAZE_SIZE - 2): 
            tqdm.write("maximum difficulty reached")
            torch.save(agent.state_dict(), f"./Models/{MODEL_NAME}.pth")
            break
        
    ###############################################
    # Recording of your agent
    ###############################################

    num_envs = 1
    ep_terminated = np.empty((STEPS_PER_UPDATE, num_envs), dtype=np.uint8)
    ep_truncated = np.empty((STEPS_PER_UPDATE, num_envs), dtype=np.uint8)
    ep_rewards_np = np.empty((STEPS_PER_UPDATE, num_envs), dtype=np.float32)

    env_video = MazeWorld(
        num_envs=num_envs, 
        maze_size = MAZE_SIZE, 
        init_nb_step_end = 200, 
        max_step = 10000, 
        n_steps_per_update = STEPS_PER_UPDATE,
        terminated = ep_terminated,
        truncated = ep_truncated,
        reward = ep_rewards_np
        )

    env_video.reset()
    frames = []

    for _ in range(10): # 10 is the nb of episodes we want to record in video

        done = False
        while not done:
            obs = torch.as_tensor(env_video._grid_arr,dtype=torch.float32, device=device)
            with torch.no_grad():
                action, _, _ = agent.get_action_value(obs)
            env_video.step(action.cpu().numpy().astype(np.int32))
            img = np.asarray(grid_to_img(env_video._grid_arr[0]))
            scale = 256//MAZE_SIZE
            img_up = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)
            frames.append(img_up.copy())
            done = ep_terminated[0,0] or ep_truncated[0,0]
    with imageio.get_writer(f'./Assets/{MODEL_NAME}_final_recording.mp4', fps=25) as writer:
        for frame in frames:
            writer.append_data(frame)

