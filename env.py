import cv2
import numpy as np
import torch
import sys
import pdb
from pathlib import Path
# sys.path.append('C:\\Projects\\payload_manipulator')
sys.path.append(str(Path(__file__).parent.parent))
# print(sys.path)
# import payload_manipulation.src.payload_env_gym
import payload_manipulation.envs
CUSTOM_ENVS = payload_manipulation.envs.CUSTOM_ENVS

GYM_ENVS = ['Pendulum-v0', 'MountainCarContinuous-v0', 'Ant-v2', 'HalfCheetah-v2', 'Hopper-v2', 'Humanoid-v2', 'HumanoidStandup-v2', 'InvertedDoublePendulum-v2', 'InvertedPendulum-v2', 'Reacher-v2', 'Swimmer-v2', 'Walker2d-v2']
CONTROL_SUITE_ENVS = ['cartpole-balance', 'cartpole-swingup', 'reacher-easy', 'finger-spin', 'cheetah-run', 'ball_in_cup-catch', 'walker-walk']
CONTROL_SUITE_ACTION_REPEATS = {'cartpole': 8, 'reacher': 4, 'finger': 2, 'cheetah': 4, 'ball_in_cup': 6, 'walker': 2}


# Preprocesses an observation inplace (from float32 Tensor [0, 255] to [-0.5, 0.5])
def preprocess_observation_(observation, bit_depth):
  observation.div_(2 ** (8 - bit_depth)).floor_().div_(2 ** bit_depth).sub_(0.5)  # Quantise to given bit depth and centre
  observation.add_(torch.rand_like(observation).div_(2 ** bit_depth))  # Dequantise (to approx. match likelihood of PDF of continuous images vs. PMF of discrete images)


# Postprocess an observation for storage (from float32 numpy array [-0.5, 0.5] to uint8 numpy array [0, 255])
def postprocess_observation(observation, bit_depth):
  return np.clip(np.floor((observation + 0.5) * 2 ** bit_depth) * 2 ** (8 - bit_depth), 0, 2 ** 8 - 1).astype(np.uint8)


def _images_to_observation(images, bit_depth):
  images = torch.tensor(cv2.resize(images, (64, 64), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1), dtype=torch.float32)  # Resize and put channel first
  # pdb.set_trace()
  preprocess_observation_(images, bit_depth)  # Quantise, centre and dequantise inplace
  return images.unsqueeze(dim=0)  # Add batch dimension

class ControlSuiteEnv():
  def __init__(self, env, symbolic, seed, max_episode_length, action_repeat, bit_depth):
    from dm_control import suite
    from dm_control.suite.wrappers import pixels
    domain, task = env.split('-')
    self.symbolic = symbolic
    self._env = suite.load(domain_name=domain, task_name=task, task_kwargs={'random': seed})
    if not symbolic:
      self._env = pixels.Wrapper(self._env)
    self.max_episode_length = max_episode_length
    self.action_repeat = action_repeat
    if action_repeat != CONTROL_SUITE_ACTION_REPEATS[domain]:
      print('Using action repeat %d; recommended action repeat for domain is %d' % (action_repeat, CONTROL_SUITE_ACTION_REPEATS[domain]))
    self.bit_depth = bit_depth

  def reset(self):
    self.t = 0  # Reset internal timer
    state = self._env.reset()
    if self.symbolic:
      return torch.tensor(np.concatenate([np.asarray([obs]) if isinstance(obs, float) else obs for obs in state.observation.values()], axis=0), dtype=torch.float32).unsqueeze(dim=0)
    else:
      return _images_to_observation(self._env.physics.render(camera_id=0), self.bit_depth)

  def step(self, action):
    action = action.detach().numpy()
    reward = 0
    for k in range(self.action_repeat):
      state = self._env.step(action)
      reward += state.reward
      self.t += 1  # Increment internal timer
      done = state.last() or self.t == self.max_episode_length
      if done:
        break
    if self.symbolic:
      observation = torch.tensor(np.concatenate([np.asarray([obs]) if isinstance(obs, float) else obs for obs in state.observation.values()], axis=0), dtype=torch.float32).unsqueeze(dim=0)
    else:
      observation = _images_to_observation(self._env.physics.render(camera_id=0), self.bit_depth)
    return observation, reward, done

  def render(self):
    cv2.imshow('screen', self._env.physics.render(camera_id=0)[:, :, ::-1])
    cv2.waitKey(1)

  def close(self):
    cv2.destroyAllWindows()
    self._env.close()

  @property
  def observation_size(self):
    return sum([(1 if len(obs.shape) == 0 else obs.shape[0]) for obs in self._env.observation_spec().values()]) if self.symbolic else (3, 64, 64)

  @property
  def action_size(self):
    return self._env.action_spec().shape[0]

  # Sample an action randomly from a uniform distribution over all valid actions
  def sample_random_action(self):
    spec = self._env.action_spec()
    return torch.from_numpy(np.random.uniform(spec.minimum, spec.maximum, spec.shape))



class GymEnv():
  def __init__(self, env, type_of_observation, seed, max_episode_length, action_repeat, bit_depth, test_env=False):
    import gym
    self.type_of_observation = type_of_observation
    self._env = gym.make(env)
    self._env.seed(seed)
    self.max_episode_length = max_episode_length
    self.action_repeat = action_repeat
    self.bit_depth = bit_depth
    try:
      self._env.set_rendering(not test_env)
    except:
      pass

  def reset(self):
    self.t = 0  # Reset internal timer
    state = self._env.reset()
    # pdb.set_trace()
    if self.type_of_observation == 'symbolic':
      return torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)
    elif self.type_of_observation == 'augmented':
      return (
          _images_to_observation(state[0], self.bit_depth),
          torch.tensor(state[1], dtype=torch.float32).unsqueeze(dim=0)
        )
    return _images_to_observation(self._env.render(mode='rgb_array'), self.bit_depth)
  
  def step(self, action):
    action = action.detach().numpy()
    reward = 0
    for k in range(self.action_repeat):
      state, reward_k, done, _ = self._env.step(action)
      reward += reward_k
      self.t += 1  # Increment internal timer
      done = done
      if done or self.t == self.max_episode_length:
        break
    if self.type_of_observation == 'symbolic':
      observation = torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)
    elif self.type_of_observation == 'augmented':
      observation = (
          _images_to_observation(state[0], self.bit_depth),
          torch.tensor(state[-1], dtype=torch.float32).unsqueeze(dim=0)
        )
    else:
      observation = _images_to_observation(self._env.render(mode='rgb_array'), self.bit_depth)
    return observation, reward, done

  def render(self):
    self._env.render()

  def close(self):
    self._env.close()

  @property
  def observation_size(self):
    if self.type_of_observation == 'symbolic':
      return self._env.observation_space.shape[0]
    elif self.type_of_observation == 'augmented':
      return self._env.observation_space.shape[0]
    return (3, 64, 64)

  @property
  def action_size(self):
    return self._env.action_space.shape[0]

  # Sample an action randomly from a uniform distribution over all valid actions
  def sample_random_action(self):
    return torch.from_numpy(self._env.action_space.sample())


def Env(env, type_of_observation, seed, max_episode_length, action_repeat, bit_depth, test_env=False):
  if env in GYM_ENVS or env in CUSTOM_ENVS:
    return GymEnv(env, type_of_observation, seed, max_episode_length, action_repeat, bit_depth, test_env=test_env)
  elif env in CONTROL_SUITE_ENVS:
    return ControlSuiteEnv(env, type_of_observation == 'symbolic', seed, max_episode_length, action_repeat, bit_depth)


# Wrapper for batching environments together
class EnvBatcher():
  def __init__(self, env_class, env_args, env_kwargs, n):
    self.n = n
    self.envs = [env_class(*env_args, **env_kwargs) for _ in range(n)]
    self.dones = [True] * n
    self.type_of_observation =  self.envs[0].type_of_observation

  # Resets every environment and returns observation
  def reset(self):
    observations = [env.reset() for env in self.envs]
    self.dones = [False] * self.n
    if self.type_of_observation == 'augmented':
      return (
        torch.cat([o[0] for o in observations]),
        torch.cat([o[1] for o in observations])
      )
    return torch.cat(observations)

 # Steps/resets every environment and returns (observation, reward, done)
  def step(self, actions):
    done_mask = torch.nonzero(torch.tensor(self.dones))[:, 0]  # Done mask to blank out observations and zero rewards for previously terminated environments
    observations, rewards, dones = zip(*[env.step(action) for env, action in zip(self.envs, actions)])
    dones = [d or prev_d for d, prev_d in zip(dones, self.dones)]  # Env should remain terminated if previously terminated
    self.dones = dones
    if self.type_of_observation == 'augmented':
      observations = (
        torch.cat([o[0] for o in observations]),
        torch.cat([o[1] for o in observations])
      )
      observations[0][done_mask] = 0
      observations[1][done_mask] = 0
    else:
      observations = torch.cat(observations)
      observations[done_mask] = 0
    rewards, dones = torch.tensor(rewards, dtype=torch.float32), torch.tensor(dones, dtype=torch.uint8)
    rewards[done_mask] = 0
    return observations, rewards, dones

  def close(self):
    [env.close() for env in self.envs]
