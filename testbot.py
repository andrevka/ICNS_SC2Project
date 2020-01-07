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
        self.score = 0
        self.pUnits = 9
        self.eUnits = 10
        self.model = Sc2Network("model.h5")
        # self.model.model.summary()

    def step(self, obs):
        super(TestAgent, self).step(obs)
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
        score_gained, self.pUnits, self.eUnits = evaluate_step(obs, self.pUnits, self.eUnits)
        self.score += score_gained
        return actions.FunctionCall(function_id, args)

    def _translateOutputToAction(self, y, avb_actions):
        f_id = 0
        a = np.argmax(y[0][0])
        args = [[int(y[2][0][0] * 9)]]
        if a == 0:
            f_id = 2
            args.append([y[1][0][0] * 79, y[1][0][1] * 64])
        elif a == 1:
            f_id = 3
            args.append([y[1][0][0] * 79, y[1][0][1] * 64])
            args.append([y[1][0][2] * 79, y[1][0][3] * 64])
        elif a == 2:
            f_id = 4
            args.append([int(y[2][0][1] * 9)])
        elif a == 3:
            f_id = 331
            args.append([y[1][0][0] * 79, y[1][0][1] * 64])
        elif a == 4:
            f_id = 333
            args.append([y[1][0][0] * 79, y[1][0][1] * 64])
        elif a == 5:
            f_id = 12
            args.append([y[1][0][0] * 79, y[1][0][1] * 64])

        if f_id not in avb_actions:
            f_id = 0

        if f_id == 0:
            args = []

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
        x.append(self.steps)
        x.append(len(marines))
        x.append(len(enemies))
        return np.asarray([x], dtype=np.dtype(np.float32))

    # Writes the score to a file
    # Resets some values to default
    def reset(self):
        super(TestAgent, self).reset()
        with open("scores.txt", "a") as f:
            f.write(str(self.score) + '\n')
        self.score = 0
        self.pUnits = 9
        self.eUnits = 10


def evaluate_step(obs, pUnits_prev, eUnits_prev):
    score_gained = 0
    marines = [unit for unit in obs.observation.feature_units
               if unit.alliance == _PLAYER_SELF]
    enemies = [unit for unit in obs.observation.feature_units
               if unit.alliance == _PLAYER_ENEMY]
    pUnits = len(marines)
    eUnits = len(enemies)
    # 5 point for every killed enemy
    # Prevents losing points for respawning enemies
    if eUnits < eUnits_prev:
        score_gained += 5 * (eUnits_prev - eUnits)
    # -1 point for every marine lost
    # Prevents gaining points for extra marines received
    if pUnits < pUnits_prev:
        score_gained += pUnits - pUnits_prev
    pUnits_prev = pUnits
    eUnits_prev = eUnits
    return score_gained, pUnits, eUnits
