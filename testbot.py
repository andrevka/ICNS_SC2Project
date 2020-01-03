from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model import Sc2Network
from data_reader import getUnitsData

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
        self.model = Sc2Network("model.h5")
        #self.model.model.summary()

    def step(self, obs):
        super(TestAgent, self).step(obs)
        self.gameloop += 1
        avb = obs.observation.available_actions

        X = self._get_unit_data(obs)
        y = self.model.predict(X)
        print(y)
        function_id, args = self._translateOutputToAction(y, avb)
        """
        function_id = np.random.choice(avb)

        args = [[np.random.randint(0, size) for size in arg.sizes]
                for arg in self.action_spec.functions[function_id].args]
        """
        print(function_id, args)
        return actions.FunctionCall(function_id, args)

    def _translateOutputToAction(self, y, avb_actions):
        f_id = 0
        a = y[0][0]

        if a[1] == 1:
            f_id = 4
            args = [[np.random.randint(0, size) for size in arg.sizes]
                    for arg in self.action_spec.functions[f_id].args]
        elif a[2] == 1:
            f_id = 331
        elif a[3] == 1:
            f_id = 333
        elif a[4] == 1:
            f_id = 567
        elif a[5] == 1:
            f_id = 12

        if f_id not in avb_actions:
            f_id = 0

        if f_id == 0:
            args = []
        elif f_id > 4:
            args = [[1], y[1][0]]
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
        print(x)
        return np.asarray([x], dtype=np.dtype(np.float32))
