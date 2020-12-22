from pysc2.lib.actions import FUNCTIONS


action_ids = [0, 1, 2, 3, 4, 6, 7, 12, 13, 42, 44, 50, 91, 183, 234, 309, 331, 332, 333, 334, 451, 452, 490]


for f in FUNCTIONS:
    if f.id in action_ids:
        # print(f)
        print("{name}={id}".format(name=f.name, id=f.id))



# for a_id in action_ids:
#     action = _FUNCTIONS[a_id]
#     print("{}={} #{}".format(action.id, a_id, action))