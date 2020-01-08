import json
import numpy as np


# Calculates the game score
def evaluate(game):
    score = 0
    pUnits_prev = 9
    eUnits_prev = 10
    for frame in game:
        score_gained, pUnits_prev, eUnits_prev = evaluate_frame(frame, pUnits_prev, eUnits_prev)
        score += score_gained
    return score


def evaluate_frame(frame, pUnits_prev, eUnits_prev):
    score_gained = 0
    pUnits = 0
    eUnits = 0
    for unit in frame['units']:
        if unit['alliance'] == 1:
            pUnits += 1
        else:
            eUnits += 1
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


# input layer
# 0-50: player units
# 50-105: enemy units
# 106: game_loop
# 107: army_count

# y1
# 0: select point (2)
# 1: select_rect (3)
# 2: select_control_group (4)
# 3: Move (331)
# 4: MovePatrol (333)
# 5: attack (12)

# y2
# 0: x1
# 1: y1
# 2: x2
# 3: y2

# y3
# n1
# n2

def get_training_data_from_file(scoreThreshold, count):
    x = []
    y = []
    y2 = []
    y3 = []
    for f in range(count):
        with open("replays/data_" + str(f) + ".txt") as json_file:
            data = json.load(json_file)

            for game in data:
                # some replays seemed to be bugged

                score = evaluate(game)
                if score <= scoreThreshold:
                    continue

                for iteration in game:
                    # OUTPUTS
                    actions = [0] * 6
                    a = iteration['fID']
                    args = []

                    if a == 2:  # select point
                        actions[0] = 1
                    elif a == 3:  # select_rect
                        actions[1] = 1
                    elif a == 4:  # select_control_group
                        actions[2] = 1
                    elif a == 331:  # Move
                        actions[3] = 1
                    elif a == 333:  # attack
                        actions[4] = 1
                    elif a == 12:  # attack
                        actions[5] = 1
                    else:
                        continue

                    a_count = iteration['army_count'] + 0.1
                    if a == 4:
                        coords = [0, 0, 0, 0]
                        args = [iteration["fArgs"][0][0] / 9, iteration["fArgs"][1][0] / 9]
                    else:
                        coords = []
                        args = [iteration["fArgs"][0][0] / 9, 0]
                        for i in iteration["fArgs"][1:]:
                            coords += i

                        while len(coords) < 4:
                            coords.append(0)

                        # normalizing
                        coords[0] = coords[0] / 79
                        coords[1] = coords[1] / 64
                        coords[2] = coords[2] / 79
                        coords[3] = coords[3] / 64

                    y3.append(args)
                    y2.append(coords)
                    y.append(actions)
                    # Inputs
                    x.append(getInputDataFromIteration(iteration))

    x = np.asarray(x)
    y = np.asarray(y)
    y2 = np.asarray(y2)
    y3 = np.asarray(y3)
    return x, y, y2, y3


def getInputDataFromIteration(iteration):
    pUnits = _getUnitsOnSide(iteration['units'], 1)
    eUnits = _getUnitsOnSide(iteration['units'], 2)
    frameInfo = getUnitsData(pUnits, 11, False) + getUnitsData(eUnits, 11, False)
    frameInfo.append(iteration["game_loop"])
    frameInfo.append(iteration["army_count"])
    frameInfo.append(iteration["zerg_count"])
    return frameInfo


# size: max number of player units. will add padding if actual number of units is less
# put_data_method: to read from file(0) or from feature_units(1)
def getUnitsData(units, size, feature_units=False):
    u = []
    for unit in units:
        if not feature_units:
            u += _putUnitDataIntoListTrain(unit)
        else:
            u += _putUnitDataIntoListInAI(unit)
    for i in range(len(u), 8 * size):
        u.append(0)
    return u


def _getUnitsOnSide(units, side):
    u = []
    for i in units:
        if i['alliance'] == side:
            u.append(i)
    return u


def _putUnitDataIntoListTrain(unit):
    # might also need to add ability x,y
    return [unit['hp'], unit['x'], unit['y'], unit['is_selected'], unit['active'], unit['type'], unit['oID0'],
            unit['oID1']]


def _putUnitDataIntoListInAI(unit):
    return [unit['health'], unit['x'], unit['y'], unit['is_selected'], unit['active'], unit['unit_type'],
            unit['order_id_0'],
            unit['order_id_1']]
