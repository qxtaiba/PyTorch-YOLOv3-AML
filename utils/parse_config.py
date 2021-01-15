import os
import numpy as np


def parse_model_cfg(path):

    # read cfg file line by line and store it
    allLines = open(path, 'r').read().split('\n')

    moduleDefinitions, validLines = [], []
    
    # extract and append all lines that are not empty and do not start with '#'
    for line in allLines:
        if line and not line.startswith("#"):
            validLines.append(line.rstrip().lstrip())

    for line in validLines:
        # check if we are at the start of a new block 
        isNewBlock = line.startswith('[')

        if isNewBlock:

            # append and populate a dictionary to moduleDefinitions
            moduleDefinitions.append({})
            moduleDefinitions[-1]['type'] = line[1:-1].rstrip()

            # check if module type is convolutional and add batch norm parameter
            if moduleDefinitions[-1]['type'] == 'convolutional':
                moduleDefinitions[-1]['batch_normalize'] = 0  # pre-populate with zeros (may be overwritten later)
        
        else:
            
            # extract key, value and strip whitepsace from key
            key, val = line.split("=")
            key = key.rstrip()

            if key == 'anchors':  # return nparray
                moduleDefinitions[-1][key] = np.array([float(x) for x in val.split(',')]).reshape((-1, 2))

            elif (key in ['from', 'layers', 'mask']):  # return array
                moduleDefinitions[-1][key] = [int(x) for x in val.split(',')]
            
            elif (key == 'size' and ',' in val): # return array
                moduleDefinitions[-1][key] = [int(x) for x in val.split(',')]

            else:
                val = val.strip()
                if val.isnumeric():
                    moduleDefinitions[-1][key] = int(val) if (int(val) - float(val)) == 0 else float(val)   # return int or float
                else:
                    moduleDefinitions[-1][key] = val  # return string

    return moduleDefinitions

def parse_data_cfg(path):
    # Parses the data configuration file
    with open(path, 'r') as f:
        lines = f.readlines()

    options = dict()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, val = line.split('=')
        options[key.strip()] = val.strip()

    return options
