import numpy as np
import random
import torch
import torch.nn as nn
import os
import inspect
import pickle
import gdown
from network import Actor


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers.
        Reference: https://github.com/MishaLaskin/rad/blob/master/curl_sac.py"""

    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


def set_seed(random_seed):
    if random_seed <= 0:
        random_seed = np.random.randint(1, 9999)
    else:
        random_seed = random_seed

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    return random_seed


def make_env(env_name, seed):
    import gymnasium as gym
    # openai gym
    env = gym.make(env_name)
    env.action_space.seed(seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = [env.action_space.low[0], env.action_space.high[0]]

    env_info = {'name': env_name, 'state_dim': state_dim, 'action_dim': action_dim, 'action_bound': action_bound, 'seed': seed}

    return env, env_info


def get_learning_info(args, seed):
    env, env_info = make_env(args.env_name, seed)
    device = 'cuda'

    alpha_dict = {'HalfCheetah-v3': args.alpha_threshold, 'Walker2d-v3': args.alpha_threshold,
                  'Ant-v3': args.alpha_threshold, 'Hopper-v3': args.alpha_threshold}

    thresholds = {"ALPHA_THRESHOLD": alpha_dict[args.env_name], "THETA_THRESHOLD": args.theta_threshold}
    max_action = 1

    t_p = Actor(env_info['state_dim'], env_info['action_dim'], (400, 300), 1)
    num_teacher_param = sum(p2.numel() for p2 in t_p.parameters())

    kwargs = {
        "env": env,
        "args": args,
        "env_info": env_info,
        "thresholds": thresholds,
        "discount": args.discount,
        "datasize": args.datasize,
        "tau": args.tau,
        "device": device,
        "num_teacher_param": num_teacher_param,
        "noise_clip": args.noise_clip * max_action,
        "policy_freq": args.policy_freq,
        "h": args.h,
    }
    return kwargs


def get_compression_ratio(num_teacher_param, agent):
    kep_w = 0
    for c in agent.actor.children():
        kep_w += c.get_num_remained_weights()
    #

    return kep_w / num_teacher_param


def load_buffer(env_name, level, datasize):
    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    file_path = os.path.join(current_dir, "teacher_buffer", "[" + level + "_buffer]_" + env_name + ".pickle")
    try:
        with open(file_path, "rb") as fr:
            buffer = pickle.load(fr)
            buffer.size = datasize
    except FileNotFoundError:
        # Download the file
        if level == 'expert':
            print("Downloading the teacher buffer...")
            if env_name == "Ant-v3":
                file_id = "10VBf3bM38bNw9WsniQvirpNjRFWp8HZO"
            elif env_name == "Walker2d-v3":
                file_id = "1ungLoqNKS4NIldZ9H2mswwGh-3Ipgy0D"
            elif env_name == "HalfCheetah-v3":
                file_id = "1wO0HwDi1GNf9d9SrDJrf9x8XMZDOTkzl"
            elif env_name == "Hopper-v3":
                file_id ="10pqCliJSM_Iyb05dxHZfYs9VlmCmPryE"
            else:
                raise ValueError("Invalid Environment Name")

            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, file_path, quiet=False)
            print("Download Complete!")
        elif level == 'medium':
            if env_name == "Ant-v3":
                file_id = "1-SKleNu6l-tY2awkx3tgVDUKbjkOaj_D"
            elif env_name == "Walker2d-v3":
                file_id = "1x6nkBBSWMRb3bENxUzcntHT1WlSNJmoh"
            elif env_name == "HalfCheetah-v3":
                file_id = "1OHkB6yVK3QcqbuJH0B_iNW_2cBnv96mR"
            elif env_name == "Hopper-v3":
                file_id ="1uqH2pgKKrhadsCXCwQWrvDvZ4ZyYFkM-"
            else:
                raise ValueError("Invalid Environment Name")

            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, file_path, quiet=False)

        else:
            raise ValueError("Invalid Level. Choose from ['expert', 'medium']")

        with open(file_path, "rb") as fr:
            buffer = pickle.load(fr)
            buffer.size = datasize

    return buffer
