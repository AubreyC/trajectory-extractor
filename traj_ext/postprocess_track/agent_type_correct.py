#################################################################################
#
# Hold agent type for a specific track. Used for manual agent_type correction
#
#################################################################################

import numpy as np
import os
import sys
import cv2
import pandas as pd
import collections

class AgentTypeCorrect(object):

    """Hold information about the agent type of a track"""
    def __init__(self, track_id, agent_type):

        self.track_id = track_id;
        self.agent_type = agent_type;

    @classmethod
    def to_csv(cls, path_to_csv, agenttype_correct_list):

        # Create empty dict
        dict_agenttype_corr= collections.OrderedDict.fromkeys(['track_id', 'agent_type']);
        dict_agenttype_corr['track_id'] = [];
        dict_agenttype_corr['agent_type'] = [];


        # Put data
        for correction in agenttype_correct_list:
            dict_agenttype_corr['track_id'].append(correction.track_id);
            dict_agenttype_corr['agent_type'].append(correction.agent_type);

        # Create dataframe
        df_agentype_corr = pd.DataFrame(dict_agenttype_corr);

        # Sort by track_id:
        df_agentype_corr.sort_values(by=['track_id'], inplace = True);

        # Write dataframe in csv
        df_agentype_corr.to_csv(path_to_csv, index=False);

        return path_to_csv;

    @classmethod
    def from_csv(cls, path_to_csv):

        # Read dataframe with panda
        df = pd.read_csv(path_to_csv);

        agenttype_correct_list = [];
        for index, row in df.iterrows():

            # print(row['track_id']);
            # print(row['agent_type']);

            track_id = int(row['track_id']);
            agent_type = str(row['agent_type']).strip()

            agenttype_correct = AgentTypeCorrect(track_id, agent_type);
            agenttype_correct_list.append(agenttype_correct);

        return agenttype_correct_list;
