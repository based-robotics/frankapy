import csv
import pickle
import argparse
import numpy as np

if __name__ == '__main__':

    input_file = '/home/saumyas/franka/frankapy/data/robot_state_data_0.txt'
    output_file = input_file[:-3]+'pkl'
    
    with open(input_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        state_dict = {}
        state_dict_index = {}
        state_dict_quantity = {}
        index = 0
        current_skill_id = 0
        current_skill_starting_time = 0
        current_skill_desc = ''
        last_skill_time = 0.0
        
        skill_time_threshold = 0.02
        
        saw_info = False
        skill_started = False
        for row in csv_reader:
            if line_count == 0:
                for item in row:
                    item_name = item[:item.find('(')]
                    state_dict_index[item_name] = index
                    state_dict_quantity[item_name] = int(item[item.find('(')+1:item.find(')')])
                    index += state_dict_quantity[item_name]
                line_count += 1
            else:
                if 'info' in row[0]:
                    saw_info = True
                    first_element = row[0]
                    current_skill_id = int(first_element[first_element.rfind(':')+2:])
                    third_element = row[2]
                    current_skill_starting_time = int(third_element[third_element.rfind(':')+2:])
                    fourth_element = row[3]
                    current_skill_desc = fourth_element[fourth_element.rfind(':')+2:]
                    state_dict[current_skill_starting_time] = {"skill_id":current_skill_id, "skill_desc": current_skill_desc, "skill_state_dict": {}}
                elif saw_info and float(row[0]) == 0:
                    skill_started = True
                    last_skill_time = 0.0
                    skill_dict = state_dict[current_skill_starting_time]
                    skill_state_dict = skill_dict["skill_state_dict"]
                    for item_name in state_dict_index.keys():
                        skill_state_dict[item_name] = []
                if skill_started and float(row[0]) - last_skill_time < skill_time_threshold:
                    skill_dict = state_dict[current_skill_starting_time]
                    skill_state_dict = skill_dict["skill_state_dict"]
                    for item_name in state_dict_index.keys():
                        starting_index = state_dict_index[item_name]
                        quantity = state_dict_quantity[item_name]
                        skill_state_dict[item_name].append([float(x) for x in row[starting_index:starting_index+quantity]])
                    line_count += 1
                    last_skill_time = float(row[0])
                elif skill_started:
                    skill_started = False
                    
                    skill_dict = state_dict[current_skill_starting_time]
                    skill_state_dict = skill_dict["skill_state_dict"]
                    for item_name in state_dict_index.keys():
                        skill_state_dict[item_name] = np.array(skill_state_dict[item_name])

    with open(output_file, 'wb') as f:
        pickle.dump(state_dict, f)