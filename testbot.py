from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from pysc2.agents import base_agent
from pysc2.lib import actions, features


_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

FUNCTIONS = actions.FUNCTIONS
RAW_FUNCTIONS = actions.RAW_FUNCTIONS

class TestAgent(base_agent.BaseAgent):

    def setup(self, obs_spec, action_spec):
        super(TestAgent, self).setup(obs_spec, action_spec)

        self.model = self.buildNN()

    def buildNN(self):
        model = Sequential()

        return model

    def step(self, obs):
        super(TestAgent, self).step(obs)
        avb = obs.observation.available_actions
        self._get_unit_data(obs)
        function_id = numpy.random.choice(avb)
        args = [[numpy.random.randint(0, size) for size in arg.sizes]
                for arg in self.action_spec.functions[function_id].args]
        print(function_id, args)
        return actions.FunctionCall(function_id, args)


    def _get_unit_data(self, obs):
        x = []

        # adding unit info
        """
        marines = [unit.tag for unit in obs.observation.raw_units
                   if unit.alliance == _PLAYER_SELF]
        roaches = [unit for unit in obs.observation.raw_units
                   if unit.alliance == _PLAYER_ENEMY]

        for i in marines:
            print(i)
        """