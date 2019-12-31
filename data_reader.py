import json
import numpy as np

def _putUnitDataIntoList(unit):
    # might also need to add ability x,y
    if len(unit['orders']) == 0:
        return [unit['health'], unit['is_active'], unit['x'], unit['y'], 0]
    return [unit['health'], unit['is_active'], unit['x'], unit['y'], unit['orders'][0]['ability_id']]


# calculates a score based on the last game_loop
# TODO: improve score calculation
def evaluate(game):
    lastIter = game[-1]
    pUnits = 0
    eUnits = 0
    for unit in lastIter['units']:
        if unit['alliance'] == 1:
            pUnits += 1
        else:
            eUnits += 1
    return 5 * (10 - eUnits) + 5 * pUnits


# input layer
# 0-50: player units
# 50-105: enemy units
# 106: game_loop
# 107: army_count
# 108: score


# 0: Do nothing
# 1: Move camera (1)
# 2: select_control_group (4)
# 3: Move (16)
# 4: MovePatrol (17)
# 5: MoveHoldPosition (18)
# 6: attack (23)

def get_training_data_from_file(file, scoreThreshold):
    with open(file) as json_file:
        data = json.load(json_file)
        x = []
        y = []
        for game in data:
            # some replays seemed to be bugged
            if len(game) == 0 or len(game[0]['units']) < 3:
                continue

            score = evaluate(game)
            if score <= scoreThreshold:
                continue

            for iteration in game:
                # INPUTS
                # adding units
                PlayerUnits = []
                EnemyUnits = []

                for unit in iteration['units']:
                    if unit['owner'] == 1:
                        PlayerUnits += _putUnitDataIntoList(unit)
                    else:
                        EnemyUnits += _putUnitDataIntoList(unit)

                # adding padding to the units
                for i in range(len(PlayerUnits), 5 * 10):
                    PlayerUnits.append(0)

                for i in range(len(EnemyUnits), 5 * 11):
                    EnemyUnits.append(0)

                # adding match info
                frameInfo = PlayerUnits + EnemyUnits
                frameInfo.append(iteration["game_loop"])
                frameInfo.append(iteration["army_count"])
                frameInfo.append(score)

                x.append(frameInfo)

                # OUTPUTS
                actions = [0] * 7
                a = iteration['actions']
                if len(a) > 0:
                    action_id = a[0]['ability_id']
                    if action_id == 0:  # Do nothing
                        actions[0] = 1
                    elif action_id == 1:  # Move camera
                        actions[1] = 1
                    elif action_id == 4:  # Select control group
                        actions[2] = 1
                    elif action_id == 16:  # Move
                        actions[3] = 1
                    elif action_id == 17:  # Move patrol
                        actions[4] = 1
                    elif action_id == 18:  # Move hold position
                        actions[5] = 1
                    elif action_id == 23:  # attack
                        actions[6] = 1
                else:
                    actions[0] = 1
                y.append(actions)
        x = np.asarray(x)
        y = np.asarray(y)
        return x, y
