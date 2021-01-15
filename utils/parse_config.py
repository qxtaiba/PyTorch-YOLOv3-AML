import os
import numpy as np


def parse_model_cfg(path):
    # Parse the yolo *.cfg file and return module definitions path may be 'cfg/yolov3.cfg', 'yolov3.cfg', or 'yolov3'

    moduleDefinitions, validLines = [], []

    # read cfg file line by line and store it
    allLines = open(path, 'r').read().split('\n')

    for line in allLines:
        
        # extract and append all lines that are not empty and do not start with '#'
        if line and not line.startswith("#"):
            
            # remove fringe whitespaces 
            validLines.append(line.rstrip().lstrip())

    for line in validLines:
        if line.startswith('['):  # This marks the start of a new block

            moduleDefinitions.append({})
            moduleDefinitions[-1]['type'] = line[1:-1].rstrip()

            if moduleDefinitions[-1]['type'] == 'convolutional':
                moduleDefinitions[-1]['batch_normalize'] = 0  # pre-populate with zeros (may be overwritten later)
        else:
            key, val = line.split("=")
            key = key.rstrip()

            if key == 'anchors':  # return nparray
                moduleDefinitions[-1][key] = np.array([float(x) for x in val.split(',')]).reshape((-1, 2))  # np anchors

            elif (key in ['from', 'layers', 'mask']) or (key == 'size' and ',' in val):  # return array
                moduleDefinitions[-1][key] = [int(x) for x in val.split(',')]

            else:
                val = val.strip()
                if val.isnumeric():  # return int or float
                    moduleDefinitions[-1][key] = int(val) if (int(val) - float(val)) == 0 else float(val)
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
