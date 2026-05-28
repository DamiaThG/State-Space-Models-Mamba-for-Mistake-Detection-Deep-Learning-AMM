'''
    helper functions to read the TSM_feature lmdb
    run this with a command line argument describing the path to the lmdb
    e.g. python read_lmdb.py TSM_features/C10095_rgb 
'''

import os
import sys
import lmdb
import numpy as np


# path to the lmdb file you want to read as a command line argument
lmdb_path = sys.argv[1]

# iterate over the entire lmdb and output all files
def extract_all_features(env):
    '''
        input:
            env: lmdb environment loaded (see main function) 
        output: a dictionary with key as the path_to_frame and value as the TSM feature (2048-D np-array)
                the lmdb key format is '{sequence_name}/{view_name}/{view_name}_{frame_no:010d}.jpg'
                e.g. nusar-2021_action_both_9011-a01_9011_user_id_2021-02-01_153724/C10095_rgb/C10095_rgb_0000000001.jpg
    '''
    # ALL THE FRAME NUMBERS ARE AT 30FPS !!!
    
    all_features = set()

    print('Iterating over the entire lmdb. This may take some time...')
    with env.begin() as e:
        cursor = e.cursor()
        
        for file, data in cursor:
            frame = file.decode("utf-8")
            data = np.frombuffer(data, dtype=np.float32)
            if data.shape[0] == 2048:
                all_features.add(frame)
            else:
                print(frame, data.shape)
            
    print(f'Features for {len(all_features)} frames loaded.')
    return all_features


# extract the feature for a particular key
def extract_by_key(env, key):
    ''' 
        input:
            env: lmdb environment loaded (see main function)
            key: the frame number in lmdb key format for which the feature is to be extracted
                 the lmdb key format is '{sequence_name}/{view_name}/{view_name}_{frame_no:010d}.jpg'
                 e.g. nusar-2021_action_both_9011-a01_9011_user_id_2021-02-01_153724/C10095_rgb/C10095_rgb_0000000001.jpg
        output: a 2048-D np-array (TSM feature corresponding to the key)
    '''
    with env.begin() as e:
        data = e.get(key.strip().encode('utf-8'))
        if data is None:
            print(f'[ERROR] Key {key} does not exist !!!')
            exit()
        data = np.frombuffer(data, 'float32')  # convert to numpy array
    return data


# main function
if __name__ == '__main__':
    # load the lmdb environment from the path
    env = lmdb.open(lmdb_path, readonly = True, lock=False)
    
    # extract_all_features() example
    all_files = extract_all_features(env)

    # extract_by_key() example
    key = 'nusar-2021_action_both_9011-a01_9011_user_id_2021-02-01_153724/C10095_rgb/C10095_rgb_0000000001.jpg'
    data = extract_by_key(env, key)
    print(data.shape)