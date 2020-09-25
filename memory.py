import numpy as np
import torch
import pdb
from env import postprocess_observation, preprocess_observation_


class ExperienceReplay():
  def __init__(self, size, type_of_observation, observation_size, action_size, bit_depth, device):
    self.device = device
    self.type_of_observation = type_of_observation
    self.size = size
    if type_of_observation == 'symbolic':
      self.observations = np.empty((size, observation_size), dtype=np.float32)
    elif type_of_observation == 'augmented':
      self.observations = dict()
      self.observations['i'] = np.empty((size, 3, 64, 64), dtype=np.uint8)
      self.observations['x'] = np.empty((size, observation_size), dtype=np.float32)
    else:
      self.observations = np.empty((size, 3, 64, 64), dtype=np.uint8)
    self.actions = np.empty((size, action_size), dtype=np.float32)
    self.rewards = np.empty((size, ), dtype=np.float32) 
    self.nonterminals = np.empty((size, 1), dtype=np.float32)
    self.idx = 0
    self.full = False  # Tracks if memory has been filled/all slots are valid
    self.steps, self.episodes = 0, 0  # Tracks how much experience has been used in total
    self.bit_depth = bit_depth

  def append(self, observation, action, reward, done):
    try:
      if self.type_of_observation == 'symbolic':
        self.observations[self.idx] = observation.numpy()
      elif self.type_of_observation == 'augmented':
        self.observations['i'][self.idx] = postprocess_observation(observation[0].numpy(), self.bit_depth)
        self.observations['x'][self.idx] = observation[1].numpy()
      else:
        self.observations[self.idx] = postprocess_observation(observation.numpy(), self.bit_depth)  # Decentre and discretise visual observations (to save memory)
    except Exception as e:
      print(e)
      pdb.set_trace()
    self.actions[self.idx] = action.numpy()
    self.rewards[self.idx] = reward
    self.nonterminals[self.idx] = not done
    self.idx = (self.idx + 1) % self.size
    self.full = self.full or self.idx == 0
    self.steps, self.episodes = self.steps + 1, self.episodes + (1 if done else 0)

  # Returns an index for a valid single sequence chunk uniformly sampled from the memory
  def _sample_idx(self, L):
    valid_idx = False
    while not valid_idx:
      idx = np.random.randint(0, self.size if self.full else self.idx - L)
      idxs = np.arange(idx, idx + L) % self.size
      valid_idx = not self.idx in idxs[1:]  # Make sure data does not cross the memory index
    return idxs

  def _retrieve_batch(self, idxs, n, L):
    vec_idxs = idxs.transpose().reshape(-1)  # Unroll indices
    if self.type_of_observation == 'augmented':
      # pdb.set_trace()
      img_obs = torch.as_tensor(self.observations['i'][vec_idxs].astype(np.float32))
      preprocess_observation_(img_obs, self.bit_depth)
      observations = (
        img_obs.reshape(L, n, 3, 64, 64),
        torch.as_tensor(self.observations['x'][vec_idxs].astype(np.float32)).reshape(L, n, *self.observations['x'].shape[1:])
      )
      return observations, self.actions[vec_idxs].reshape(L, n, -1), self.rewards[vec_idxs].reshape(L, n), self.nonterminals[vec_idxs].reshape(L, n, 1)

    observations = torch.as_tensor(self.observations[vec_idxs].astype(np.float32))
    if not self.type_of_observation == 'symbolic':
      preprocess_observation_(observations, self.bit_depth)  # Undo discretisation for visual observations
    return observations.reshape(L, n, *observations.shape[1:]), self.actions[vec_idxs].reshape(L, n, -1), self.rewards[vec_idxs].reshape(L, n), self.nonterminals[vec_idxs].reshape(L, n, 1)

  # Returns a batch of sequence chunks uniformly sampled from the memory
  def sample(self, n, L):
    batch = self._retrieve_batch(np.asarray([self._sample_idx(L) for _ in range(n)]), n, L)
    if not self.type_of_observation == 'augmented':
      return [torch.as_tensor(item).to(device=self.device) for item in batch]
    return [
      (torch.as_tensor(batch[0][0]).to(device=self.device),
      torch.as_tensor(batch[0][1]).to(device=self.device)),
      torch.as_tensor(batch[1]).to(device=self.device),
      torch.as_tensor(batch[2]).to(device=self.device),
      torch.as_tensor(batch[3]).to(device=self.device),
    ]
