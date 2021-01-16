import os
import numpy as np

def parseModel(path):
    # init empty lists
    moduleDefinitions, validLines = [], []
    # read cfg file line by line and store it
    allLines = open(path, 'r').read().split('\n')
    
    for line in allLines:
        # check if line is not empty and do not start with '#'
        if line and not line.startswith("#"):
            # append line and strip all fringe whitespace 
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
                # pre-populate with zeros (may be overwritten later)
                moduleDefinitions[-1]['batch_normalize'] = 0  
        
        else:
            # extract key, value pair
            key, val = line.split("=")
            # strip whitespace 
            key = key.rstrip()

            # return a numpy array 
            if key == 'anchors':  
                moduleDefinitions[-1][key] = np.array([float(x) for x in val.split(',')]).reshape((-1, 2))
            # return a regular array 
            elif (key in ['from', 'layers', 'mask']):  
                moduleDefinitions[-1][key] = [int(x) for x in val.split(',')]
            # return a regular array 
            elif (key == 'size' and ',' in val): 
                moduleDefinitions[-1][key] = [int(x) for x in val.split(',')]

            else:
                # strip whitespace 
                val = val.strip()
                # return int/float 
                if val.isnumeric():
                    moduleDefinitions[-1][key] = int(val) if (int(val) - float(val)) == 0 else float(val)   # return int or float
                # return string 
                else:
                    moduleDefinitions[-1][key] = val  

    return moduleDefinitions

def parseData(path):
    # init output dictionary 
    options = dict()

    # open are read data file into lines 
    with open(path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        # strip whitespace 
        line = line.strip()
        # check if line is empty or starts with a '#' (indicates a comment)
        if line == '' or line.startswith('#'): continue
        # extract key, value pair 
        key, val = line.split('=')
        # add key,value pair to dictionary 
        options[key.strip()] = val.strip()

    return options
