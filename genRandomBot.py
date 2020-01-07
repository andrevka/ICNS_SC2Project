from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import json

from pysc2.agents import base_agent
from pysc2.lib import actions, features

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

FUNCTIONS = actions.FUNCTIONS
RAW_FUNCTIONS = actions.RAW_FUNCTIONS


class GenRandomAgent(base_agent.BaseAgent):

    def setup(self, obs_spec, action_spec):
        super(GenRandomAgent, self).setup(obs_spec, action_spec)

        if "feature_units" not in obs_spec:
            raise Exception(
                "This agent requires the feature_units observation. Use flag '--use_feature_units' to enable feature units")
        self.writeEvery = 1000
        self.fileN = -1
        self.newFile()

    def step(self, obs):
        super(GenRandomAgent, self).step(obs)
        function_id = numpy.random.choice(obs.observation.available_actions)
        args = [[numpy.random.randint(0, size) for size in arg.sizes]
                for arg in self.action_spec.functions[function_id].args]
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

    def reset(self):
        super(GenRandomAgent, self).reset()
        if self.episodes == 0:
            return

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
