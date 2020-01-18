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
    x2 = [[], [], [], [], [], []]
    y2 = [[], [], [], [], [], []]
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
                    k = 0
                    if a == 2:  # select point
                        actions[k] = 1
                        y2[k].append([iteration["fArgs"][0][0] / 9, iteration["fArgs"][1][0] / 79,
                                      iteration["fArgs"][1][1] / 64])
                    elif a == 3:  # select_rect
                        k = 1
                        actions[k] = 1
                        y2[k].append([iteration["fArgs"][0][0] / 9, iteration["fArgs"][1][0] / 79,
                                      iteration["fArgs"][1][1] / 64, iteration["fArgs"][2][0] / 79,
                                      iteration["fArgs"][2][1] / 64])
                    elif a == 4:  # select_control_group
                        k = 2
                        actions[k] = 1
                        y2[k].append([iteration["fArgs"][0][0] / 9, iteration["fArgs"][1][0] / 9])
                    elif a == 331:  # Move
                        k = 3
                        actions[k] = 1
                        y2[k].append([iteration["fArgs"][0][0] / 9, iteration["fArgs"][1][0] / 79,
                                      iteration["fArgs"][1][1] / 64])
                    elif a == 333:  # attack
                        k = 4
                        actions[k] = 1
                        y2[k].append([iteration["fArgs"][0][0] / 9, iteration["fArgs"][1][0] / 79,
                                      iteration["fArgs"][1][1] / 64])
                    elif a == 12:  # attack
                        k = 5
                        actions[k] = 1
                        y2[k].append([iteration["fArgs"][0][0] / 9, iteration["fArgs"][1][0] / 79,
                                      iteration["fArgs"][1][1] / 64])
                    else:
                        continue

                    y.append(actions)
                    # Inputs
                    x.append(np.asarray(getInputDataFromIteration(iteration)))
                    x2[k].append(np.asarray(getInputDataFromIteration(iteration)))

    x = np.asarray(x)
    y = np.asarray(y)
    x2 = np.asarray(x2)
    y2 = np.asarray(y2)

    return x, y, x2, y2


def getInputDataFromIteration(iteration):
    pUnits = _getUnitsOnSide(iteration['units'], 1)
    eUnits = _getUnitsOnSide(iteration['units'], 4)
    frameInfo = getUnitsData(pUnits, 9, False) + getUnitsData(eUnits, 10, False)
    frameInfo.append(iteration["game_loop"])
    frameInfo.append(iteration["army_count"])
    frameInfo.append(iteration["zerg_count"])
    return frameInfo


# size: max number of player units. will add padding if actual number of units is less
# put_data_method: to read from file(0) or from feature_units(1)
def getUnitsData(units, size, feature_units=False):
    u = []
    i = 0
    for unit in units:
        if not feature_units:
            u += _putUnitDataIntoListTrain(unit)
        else:
            u += _putUnitDataIntoListInAI(unit)
        i += 1
        if i >= size:
            break

    for i in range(len(u), 8 * size):
        u.append(0)

    return u


def _getUnitsOnSide(units, side):
    u = []
    for i in units:
        if i['alliance'] == side:
            u.append(i)
        if i['alliance'] == 3 or i['alliance'] == 2:
            print("WARNING! some units have some other alliance")

    return u


def _putUnitDataIntoListTrain(unit):
    # might also need to add ability x,y
    return [unit['hp'], unit['x'], unit['y'], unit['is_selected'], unit['active'], unit['type'], unit['oID0'],
            unit['oID1']]


def _putUnitDataIntoListInAI(unit):
    return [unit['health'], unit['x'], unit['y'], unit['is_selected'], unit['active'], unit['unit_type'],
            unit['order_id_0'],
            unit['order_id_1']]
