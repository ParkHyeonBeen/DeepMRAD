import numpy as np
import pandas as pd
import gym
import random, math
import torch
import torch.nn as nn
from collections import deque
from skimage.util.shape import view_as_windows
import matplotlib.pyplot as plt


# For testing whether a number is close to zero
_FLOAT_EPS = np.finfo(np.float64).eps
_EPS4 = _FLOAT_EPS * 4.0

data = None
path_data = np.empty([1, 3])

## related to control ##
def quat2mat(quat):
    """ Convert Quaternion to Rotation matrix.  See rotation.py for notes """
    quat = np.asarray(quat, dtype=np.float64)
    assert quat.shape[-1] == 4, "Invalid shape quat {}".format(quat)

    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    Nq = np.sum(quat * quat, axis=-1)
    s = 2.0 / Nq
    X, Y, Z = x * s, y * s, z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ, zZ = y * Y, y * Z, z * Z

    mat = np.empty(quat.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 0, 0] = 1.0 - (yY + zZ)
    mat[..., 0, 1] = xY - wZ
    mat[..., 0, 2] = xZ + wY
    mat[..., 1, 0] = xY + wZ
    mat[..., 1, 1] = 1.0 - (xX + zZ)
    mat[..., 1, 2] = yZ - wX
    mat[..., 2, 0] = xZ - wY
    mat[..., 2, 1] = yZ + wX
    mat[..., 2, 2] = 1.0 - (xX + yY)
    return np.where((Nq > _FLOAT_EPS)[..., np.newaxis, np.newaxis], mat, np.eye(3))


def rot2rpy(R):
    temp = np.array([0, 0, 1]) @ R
    pitch = math.asin(-temp[0])
    roll = math.asin(temp[1] / math.cos(pitch))
    yaw = math.acos(R[0, 0] / math.cos(pitch))

    return roll, pitch, yaw


def quat2rpy(quat):
    R = quat2mat(quat)
    euler = rot2rpy(R)
    euler = np.array(euler)
    return euler

def denomalize(input, act_max, act_min):
    denormal_mat = np.zeros((len(input), len(input)))
    np.fill_diagonal(denormal_mat, (act_max - act_min) / 2)
    denormal_bias = (act_max + act_min) / 2
    input = input @ denormal_mat + denormal_bias
    return input

def add_noise(action, percent = 0.1):
    for i in range(len(action)):
        action[i] += action[i]*random.uniform(-percent, percent)
    return action

def add_disturbance(action, step, terminal_time, percent = 0.1):
    for i in range(len(action)):
        action[i] += action[i]*percent*math.sin((random.choice([2, 4, 8])*math.pi / terminal_time)*step)
    return action

## related to saved data ##

def init_data():
    global data
    data = None

def put_data(obs):
    global data
    if data is None:
        data = obs
    else:
        data = np.vstack((data, obs))

def put_path(obs):
    global path_data
    path_data = np.vstack((path_data, obs[:3]))

def plot_data(obs, label=None):
    global data
    print(data)
    if data is None:
        data = obs
    else:
        data = np.vstack((data, obs))
    if label is None:
        plt.plot(data)
    else:
        plt.plot(data, label=label)
        plt.legend()
    plt.show(block=False)
    plt.pause(0.0001)
    plt.cla()

def plot_path(obs, label=None):
    global path_data
    path_data = np.vstack((path_data, obs[:2]))
    if label is None:
        plt.plot(path_data[:,0], path_data[:,1])
    else:
        plt.plot(path_data[:, 0], path_data[:, 1], label=label)
        plt.legend()
    plt.show(block=False)
    plt.pause(0.0001)
    plt.cla()

def save_data(path, fname):
    global data
    df = pd.DataFrame(data)
    df.to_csv(path + fname + ".csv")

def save_path(path, name):
    global path_data
    df = pd.DataFrame(path_data)
    df.to_csv(path + name + ".csv")

def sava_network(network, fname : str, root : str):
    if "policy" in fname:
        torch.save(network.state_dict(), root + "saved_net/policy/" + fname)
    elif "model" in fname:
        if "DNN" in fname:
            torch.save(network.state_dict(), root + "saved_net/model/DNN/" + fname)
        elif "BNN" in fname:
            torch.save(network.state_dict(), root + "saved_net/model/BNN/" + fname)
        else:
            torch.save(network.state_dict(), root + "saved_net/model/Etc/" + fname)
    else:
        raise Exception(" check your file name ")

## related to gym

def set_seed(random_seed):
    if random_seed <= 0:
        random_seed = np.random.randint(1, 9999)
    else:
        random_seed = random_seed

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    random.seed(random_seed)

    return random_seed

def gym_env(env_name, random_seed):
    import gym
    # openai gym
    env = gym.make(env_name)
    env.seed(random_seed)
    env.action_space.seed(random_seed)

    test_env = gym.make(env_name)
    test_env.seed(random_seed)
    test_env.action_space.seed(random_seed)

    return env, test_env

def soft_update(network, target_network, tau):
    for param, target_param in zip(network.parameters(), target_network.parameters()):
        target_param.data.copy_(target_param.data * (1 - tau) + param.data * tau)

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

def copy_weight(network, target_network):
    target_network.load_state_dict(network.state_dict())

def tie_conv(source, target):
    #source ->  target
    for i in range(source.layer_num):
        target.conv[i].weight = source.conv[i].weight
        target.conv[i].bias = source.conv[i].bias

def atari_env(env_name, image_size, frame_stack, frame_skip, random_seed):
    import gym
    from gym.wrappers import AtariPreprocessing, FrameStack
    env = gym.make(env_name)
    env = AtariPreprocessing(env, frame_skip=frame_skip, screen_size=image_size, grayscale_newaxis=True)
    env = FrameStack(env, frame_stack)

    env._max_episode_steps = 10000
    env.seed(random_seed)
    env.action_space.seed(random_seed)

    test_env = gym.make(env_name)
    test_env = AtariPreprocessing(test_env, frame_skip=frame_skip, screen_size=image_size,
                                  grayscale_newaxis=True)
    test_env._max_episode_steps = 10000
    test_env = FrameStack(test_env, frame_stack)
    test_env.seed(random_seed)
    test_env.action_space.seed(random_seed)

    return env, test_env

def dmc_env(env_name, random_seed):
    import dmc2gym
    # deepmind control suite
    domain_name = env_name.split('/')[0]
    task_name = env_name.split('/')[1]
    env = dmc2gym.make(domain_name=domain_name, task_name=task_name, seed=random_seed)
    test_env = dmc2gym.make(domain_name=domain_name, task_name=task_name, seed=random_seed)

    return env, test_env

def dmc_image_env(env_name, image_size, frame_stack, frame_skip, random_seed):
    import dmc2gym
    domain_name = env_name.split('/')[0]
    task_name = env_name.split('/')[1]
    env = dmc2gym.make(domain_name=domain_name, task_name=task_name, seed=random_seed, visualize_reward=False,
                       from_pixels=True, height=image_size, width=image_size,
                       frame_skip=frame_skip)  # Pre image size for curl, image size for dbc
    env = FrameStack(env, k=frame_stack)

    test_env = dmc2gym.make(domain_name=domain_name, task_name=task_name, seed=random_seed, visualize_reward=False, from_pixels=True, height=image_size, width=image_size,
                            frame_skip=frame_skip)  # Pre image size for curl, image size for dbc
    test_env = FrameStack(test_env, k=frame_stack)

    return env, test_env

def dmcr_env(env_name, image_size, frame_skip, random_seed, mode='classic'):
    assert mode in {'classic', 'generalization', 'sim2real'}

    import dmc_remastered as dmcr

    domain_name = env_name.split('/')[0]
    task_name = env_name.split('/')[1]
    if mode == 'classic':#loads a training and testing environment that have the same visual seed
        env, test_env = dmcr.benchmarks.classic(domain_name, task_name, visual_seed=random_seed, width=image_size, height=image_size, frame_skip=frame_skip)
    elif mode == 'generalization':#creates a training environment that selects a new visual seed from a pre-set range after every reset(), while the testing environment samples from visual seeds 1-1,000,000
        env, test_env = dmcr.benchmarks.visual_generalization(domain_name, task_name, num_levels=100, width=image_size, height=image_size, frame_skip=frame_skip)
    elif mode == 'sim2real':#approximates the challenge of transferring control policies from simulation to the real world by measuring how many distinct training levels the agent needs access to before it can succesfully operate in the original DMC visuals that it has never encountered.
        env, test_env = dmcr.benchmarks.visual_sim2real(domain_name, task_name, num_levels=random_seed, width=image_size, height=image_size, frame_skip=frame_skip)

    return env, test_env

def procgen_env(env_name, frame_stack, random_seed):
    import gym
    env_name = "procgen:procgen-{}-v0".format(env_name)
    env = gym.make(env_name, render_mode='rgb_array')
    env._max_episode_steps = 1000
    env = FrameStack(env, frame_stack, data_format='channels_last')

    test_env = gym.make(env_name, render_mode='rgb_array')
    test_env._max_episode_steps = 1000
    test_env = FrameStack(test_env, frame_stack, data_format='channels_last')

    return env, test_env

def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2**bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2**(8 - bits))
    obs = obs / bins
    obs = obs + torch.rand(size=obs.shape) / bins
    obs = obs - 0.5
    return obs

def random_crop(imgs, output_size, data_format='channels_first'):#random crop for curl
    """
    Vectorized way to do random crop using sliding windows
    and picking out random ones

    args:
        imgs, batch images with shape (B,C,H,W)
    """
    # batch size
    n = imgs.shape[0]
    img_size = imgs.shape[-1]
    crop_max = img_size - output_size

    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    # creates all sliding windows combinations of size (output_size)
    windows = view_as_windows(
        imgs, (1, output_size, output_size, 1))[..., 0,:,:, 0]
    # selects a random window for each batch element
    cropped_imgs = windows[np.arange(n), w1, h1]
    return cropped_imgs

def center_crop_image(image, output_size):#center crop for curl
    h, w = image.shape[1:]
    new_h, new_w = output_size, output_size

    top = (h - new_h)//2
    left = (w - new_w)//2

    image = image[:, top:top + new_h, left:left + new_w]

    return image

def center_crop_images(image, output_size):
    h, w = image.shape[2:]
    new_h, new_w = output_size, output_size

    top = (h - new_h)//2
    left = (w - new_w)//2

    image = image[:, :, top:top + new_h, left:left + new_w]
    return image


class FrameStack(gym.Wrapper):
    def __init__(self, env, k, data_format='channels_first'):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape

        if data_format == 'channels_first':
            self.observation_space = gym.spaces.Box(
                low=0,
                high=1,
                shape=((shp[0] * k,) + shp[1:]),
                dtype=env.observation_space.dtype
            )
            self.channel_first = True
        else:
            self.observation_space = gym.spaces.Box(
                low=0,
                high=1,
                shape=(shp[0:-1] + (shp[-1] * k,)),
                dtype=env.observation_space.dtype
            )
            self.channel_first = False

        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        if self.channel_first == True:
            return np.concatenate(list(self._frames), axis=0)
        elif self.channel_first == False:
            return np.concatenate(list(self._frames), axis=-1)

