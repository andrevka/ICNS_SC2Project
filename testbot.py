from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model import Sc2Network
from data_reader import getUnitsData

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

from pysc2.agents import base_agent
from pysc2.lib import actions, features

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

FUNCTIONS = actions.FUNCTIONS
RAW_FUNCTIONS = actions.RAW_FUNCTIONS


def _xy_locs(mask):
    """Mask should be a set of bools from comparison with a feature layer."""
    y, x = mask.nonzero()
    return list(zip(x, y))


class TestAgent(base_agent.BaseAgent):

    def setup(self, obs_spec, action_spec):
        super(TestAgent, self).setup(obs_spec, action_spec)

        if "feature_units" not in obs_spec:
            raise Exception(
                "This agent requires the feature_units observation. Use flag '--use_feature_units' to enable feature units")
        self.gameloop = 0
        self.model = Sc2Network("model")
        self.model.model.summary()


    def step(self, obs):
        super(TestAgent, self).step(obs)
        self.gameloop += 1
        avb = obs.observation.available_actions
        X = self._get_unit_data(obs)
        function_id, args = self._translateOutputToAction(self.model.predict(X)[0])
        """
        function_id = np.random.choice(avb)
        
        args = [[np.random.randint(0, size) for size in arg.sizes]
                for arg in self.action_spec.functions[function_id].args]
                """
        print(function_id, args)
        return actions.FunctionCall(function_id, args)

    def _translateOutputToAction(self, y):
        f_id = 0
        if y[1] == 1:
            f_id = 1
        elif y[2] == 1:
            f_id = 4
        elif y[3] == 1:
            f_id = 16
        elif y[4] == 1:
            f_id = 17
        elif y[5] == 1:
            f_id = 18
        elif y[6] == 1:
            f_id = 23

        args = [[np.random.randint(0, size) for size in arg.sizes]
                for arg in self.action_spec.functions[f_id].args]

        return f_id, args

    def _get_unit_data(self, obs):
        x = []
        # adding unit info
        print("--------------------------------")
        marines = [unit for unit in obs.observation.feature_units
                   if unit.alliance == _PLAYER_SELF]
        enemies = [unit for unit in obs.observation.feature_units
                   if unit.alliance == _PLAYER_ENEMY]

        x += getUnitsData(marines, 10, True)
        x += getUnitsData(enemies, 11, True)
        x.append(self.gameloop)
        x.append(len(marines))
        return np.asarray([x], dtype=np.dtype(np.float32))

