import numpy as np
from gym import Wrapper


class EnvWrapper(Wrapper):
    """Observation wrapper that stacks the observations in a rolling manner.
    """

    def __init__(self, env, without_oracel=False):
        super(EnvWrapper, self).__init__(env)
        self.env = env
        self.cutins = True
        self.without_oracel = without_oracel

    def step(self, action):
        observation, reward, done, info = self.env.step(action-10)
        if observation is None:
            return None, 0, True, None
        info = {"cutin": observation['did_cutin']}
        state = self._create_next_obs(observation)

        return state, reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        state = self._create_next_obs(observation)
        return state

    def _create_next_obs(self, state):
        """ vector state
        pos 0 : ego velocity
        pos 1: ego acc
        pos 2: ego jerk
        pos 3: target pos
        pos 4: target vel
        pos 5: acc


        :param state:
        :return:
        """
        state_list = []
        for k in state.keys():
            if self.without_oracel:
                if k == "oracle_preds":
                    continue
            tmp = np.array(state[k])
            if len(tmp.shape) > 1:
                tmp = tmp.squeeze(0)
            state_list.append(tmp)
        state = np.concatenate(list(state_list), 0)
        return state


def write_into_file(pathname, text):
    """
    """
    with open(pathname+".txt", "a") as myfile:
        myfile.write(text)
        myfile.write('\n')