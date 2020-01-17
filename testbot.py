from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model import Sc2Network
from data_reader import getUnitsData

import numpy as np
import json

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

# TODO: rename bot
class TestAgent(base_agent.BaseAgent):

    def setup(self, obs_spec, action_spec):
        super(TestAgent, self).setup(obs_spec, action_spec)

        if "feature_units" not in obs_spec:
            raise Exception(
                "This agent requires the feature_units observation. Use flag '--use_feature_units' to enable feature units")
        self.score = 0
        self.pUnits = 9
        self.eUnits = 10
        self.writeEvery = 200
        self.fileN = -1
        self.newFile()
        self.model = Sc2Network("model")
        print("Setup done!")
        # self.model.model.summary()

    def step(self, obs):
        super(TestAgent, self).step(obs)
        avb = obs.observation.available_actions

        X = self._get_unit_data(obs)
        y = self.model.predict(X)
        function_id, args = self._translateOutputToAction(y, avb)
        """
        function_id = np.random.choice(avb)
        args = [[np.random.randint(0, size) for size in arg.sizes]
                for arg in self.action_spec.functions[function_id].args]
        """
        #print(function_id, args)
        score_gained, self.pUnits, self.eUnits = evaluate_step(obs, self.pUnits, self.eUnits)
        self.score += score_gained
        self.saveStep(obs, function_id, args)
        return actions.FunctionCall(function_id, args)
    
    def saveStep(self, obs, fID, args):

        s = {"fID": int(fID), "fArgs": args}
        units = []
        for i in obs.observation.feature_units:
            units.append(
                {"hp": int(i["health"]), "x": int(i["x"]), "y": int(i["y"]), "alliance": int(i["alliance"]),
                 "type": int(i['unit_type']),
                 "is_selected": int(i["is_selected"]), "oID0": int(i["order_id_0"]), "oID1": int(i["order_id_1"]),
                 "active": int(i["active"])})
        s["units"] = units
        s["game_loop"] = int(self.steps)
        marines = len([unit for unit in obs.observation.feature_units
                       if unit.alliance == _PLAYER_SELF])
        enemies = len([unit for unit in obs.observation.feature_units
                       if unit.alliance == _PLAYER_ENEMY])
        s["army_count"] = marines
        s["zerg_count"] = enemies
        self.loops.append(s)

    def _translateOutputToAction(self, y, avb_actions):
        f_id, args = y

        if f_id not in avb_actions:
            f_id = 0

        if f_id == 0:
            args = []
            return f_id, args

        mSize = self.action_spec.functions[f_id].args[0].sizes[0]
        if args[0][0] >= mSize:
            args[0][0] = np.random.randint(0, mSize)

        return f_id, args

    def _get_unit_data(self, obs):
        x = []
        # adding unit info
        #print("--------------------------------")
        marines = [unit for unit in obs.observation.feature_units
                   if unit.alliance == _PLAYER_SELF]
        enemies = [unit for unit in obs.observation.feature_units
                   if unit.alliance == _PLAYER_ENEMY]

        x += getUnitsData(marines, 9, True)
        x += getUnitsData(enemies, 10, True)
        x.append(self.steps)
        x.append(len(marines))
        x.append(len(enemies))
        return np.asarray([x], dtype=np.dtype(np.float32))

    # Writes the score to a file
    # Resets some values to default
    def reset(self):
        super(TestAgent, self).reset()
        with open("scores_test.txt", "a") as f:
            f.write(str(self.score) + '\n')
        self.score = 0
        self.pUnits = 9
        self.eUnits = 10
        self.games.append(self.loops)
        self.loops = []
        if self.episodes % self.writeEvery == 0:
            self.writeToFile()
            self.newFile()
            
    def newFile(self):
        self.loops = []
        self.games = []
        self.fileN += 1

    def writeToFile(self):
        with open("data_" + str(self.fileN) + ".txt", "w") as outfile:
            json.dump(self.games, outfile)


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
    return score_gained, pUnits, eUnits