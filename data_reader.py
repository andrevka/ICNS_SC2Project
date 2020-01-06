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
        #5 point for every killed enemy
        #Prevents losing points for respawning enemies
        if eUnits < eUnits_prev:
            score_gained += 5*(eUnits_prev - eUnits)
        #-1 point for every marine lost
        #Prevents gaining points for extra marines received
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


# 0: Do nothing
# 1: select_control_group (4)
# 2: Move (16)
# 3: MovePatrol (17)
# 4: MoveHoldPosition (18)
# 5: attack (23)

def get_training_data_from_file(file, scoreThreshold):
    with open(file) as json_file:
        data = json.load(json_file)
        x = []
        y = []
        y2 = []
        for game in data:
            # some replays seemed to be bugged
            if len(game) == 0 or len(game[0]['units']) < 3:
                continue

            score = evaluate(game)
            if score <= scoreThreshold:
                continue

            for iteration in game:
                # OUTPUTS
                actions = [0] * 5
                a = iteration['actions']
                if len(a) > 0:
                    action_id = a[0]['ability_id']
                    if action_id == 4:  # Select control group
                        actions[0] = 1
                    elif action_id == 16:  # Move
                        actions[1] = 1
                    elif action_id == 17:  # Move patrol
                        actions[2] = 1
                    elif action_id == 18:  # Move hold position
                        actions[3] = 1
                    elif action_id == 23:  # attack
                        actions[4] = 1
                    else:
                        continue
                    y2.append(np.asarray([a[0]['x']/44, a[0]['y']/36], dtype=np.dtype(np.float32)))

                else:
                    continue
                y.append(actions)
                # Inputs
                x.append(getInputDataFromIteration(iteration))

        x = np.asarray(x)
        y = np.asarray(y)
        return x, y, y2


def getInputDataFromIteration(iteration):
    pUnits = _getUnitsOnSide(iteration['units'],1)
    eUnits = _getUnitsOnSide(iteration['units'],2)
    frameInfo = getUnitsData(pUnits, 10, False) + getUnitsData(eUnits, 11, False)
    frameInfo.append(iteration["game_loop"])
    frameInfo.append(iteration["army_count"])
    return frameInfo

# size: max number of player units. will add padding if actual number of units is less
# put_data_method: to read from file(0) or from feature_units(1)
def getUnitsData(units, size, feature_units = False):
    u = []
    for unit in units:
        if not feature_units:
            u += _putUnitDataIntoListTrain(unit)
        else:
            u += _putUnitDataIntoListInAI(unit)
    for i in range(len(u), 5 * size):
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
    if len(unit['orders']) == 0:
        return [unit['health'], unit['is_active'], unit['x'], unit['y'], 0]
    return [unit['health'], unit['is_active'], unit['x'], unit['y'], unit['orders'][0]['ability_id']]

def _putUnitDataIntoListInAI(unit):
    if unit['order_length'] == 0:
        return [unit['health'], unit['active'], unit['x'], unit['y'], 0]
    return [unit['health'], unit['active'], unit['x'], unit['y'], unit['order_id_0']]