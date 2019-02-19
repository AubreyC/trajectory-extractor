# -*- coding: utf-8 -*-

##########################################################################################
#
# EVALUATE THE TRACKIGN COVERAGE
#
# Proportion of the track actually being tracked by the tracker in % of the true
# trajectory length.
#
##########################################################################################

import numpy as np
import pandas
import matplotlib.pyplot as plt
import argparse


class TrajResult():
    def __init__(self, oid, total_length, nb_match, nb_switch, nb_miss):
        self.oid = oid;
        self.total_length = total_length;
        self.nb_match = nb_match;
        self.nb_switch = nb_switch;
        self.nb_miss = nb_miss;

        self.percent_miss = float(self.nb_miss)/float(self.total_length);
        self.percent_match = 1-self.percent_miss;


def find_real_switch(data):

    print('\n# Find real switch:\n')
    real_switch = 0;

    bool_switch = data['Type'] == 'SWITCH';
    data_switch = data[bool_switch];

    oid_switch = data_switch['OId']
    for oid in oid_switch:

        print('\n\nSwitch OId: {}'.format(oid))


        data_oid_bool = (data['OId'] == int(oid));
        data_match_bool = (data['Type'] == 'MATCH');
        data_oid__match_bool  = data_oid_bool* data_match_bool;
        data_oid_match = data[data_oid__match_bool];


        data_hid = data_oid_match['HId'];
        data_hid = data_hid.drop_duplicates();


        for hid in data_hid:
            print('Matched HId: {}'.format(hid));

            data_hid_bool = (data['HId'] == int(hid));
            data_match_bool = (data['Type'] == 'MATCH');
            data_hid_match_bool = data_hid_bool* data_match_bool;
            data_hid_match = data[data_hid_match_bool];

            data_oid = data_hid_match['OId'];
            data_oid = data_oid.drop_duplicates();

            print('Data oid size {}'.format(data_oid.size))

            if data_oid.size > 1:
                real_switch = real_switch + 1;

            for oid_match in data_oid:
                print('HId: {} matches OId: {}'.format(hid, oid_match))

        print('\n\nReal Switch: {}'.format(real_switch))


def compute_tracking_coverage(data, display_hist = False):
    print('\n# Compute tracking coverage:\n')

    traj_oid_list= [];

    data_oid = data['OId'].drop_duplicates();

    for oid in data_oid:

        if not np.isnan(oid):
            data_oid_bool = (data['OId'] == int(oid));

            data_match_bool = (data['Type'] == 'MATCH');
            data_oid_match_bool  = data_oid_bool* data_match_bool;
            data_oid_match = data[data_oid_match_bool];
            nb_match = data_oid_match.size;

            data_miss_bool = (data['Type'] == 'MISS');
            data_oid_miss_bool  = data_oid_bool* data_miss_bool;
            data_oid_miss = data[data_oid_miss_bool];
            nb_miss = data_oid_miss.size;

            data_switch_bool = (data['Type'] == 'SWITCH');
            data_oid_switch_bool  = data_oid_bool* data_switch_bool;
            data_oid_switch = data[data_oid_switch_bool];
            nb_switch = data_oid_switch.size;

            total_length = nb_switch + nb_miss + nb_match;

            traj_res = TrajResult(oid, total_length, nb_match, nb_switch, nb_miss);

            traj_oid_list.append(traj_res);

    mostly_missed = mostly_tracked = partially_tracked = 0;

    pecrcent_miss_list = [0.1,0.2,0.3,0.4];
    pecrcent_miss_traj = [0.1,0.2,0.3,0.4];


    percent_track_list = [];

    for traj_res in traj_oid_list:

        percent_track_list.append(1-traj_res.percent_miss);

        if traj_res.percent_miss < 0.2:
            mostly_tracked = mostly_tracked + 1;

        elif traj_res.percent_miss < 0.8:
            partially_tracked = partially_tracked + 1;
            # print('Track oid: {} total length: {} miss: {} {}%'.format(oid, total_length, nb_miss, percent_miss))

        elif traj_res.percent_miss > 0.8:
            mostly_missed = mostly_missed + 1;

    print('mostly_missed (<20%): {}'.format(mostly_missed))
    print('partially_tracked (80%< <20%): {}'.format(partially_tracked))
    print('mostly_tracked (>20%): {}'.format(mostly_tracked))

    hist_np = np.histogram(percent_track_list,  bins=[0, 0.2, 0.7, 0.9, 1.0])
    print('\nHistogram:\n Bins: {}\n Traj number: {}\n Traj percent: {}\n'.format(hist_np[1],hist_np[0], 100*(hist_np[0]/len(traj_oid_list))));

    if display_hist:
        plt.figure('Hist xy')
        plt.hist(percent_track_list, weights=100*np.ones(len(percent_track_list)) / len(percent_track_list), bins=10);
        plt.show()


def main():

    # Print instructions
    print("############################################################")
    print("Tracking coverage analysis")
    print("############################################################\n")

    ##########################################################
    # Parse Arguments
    ##########################################################
    argparser = argparse.ArgumentParser(
        description='Camera Calibration Manual Mode')
    argparser.add_argument(
        '-i', '--input',
        help='MOT panda dataframe (csv) output after trajectory accuracy evaluation',
        required=True);
    argparser.add_argument(
        '-s', '--show',
        help='MOT panda dataframe (csv) output after trajectory accuracy evaluation',
        action='store_true');

    args = argparser.parse_args();

    # Read data
    data = pandas.read_csv(args.input);

    # Find real switch
    find_real_switch(data);

    # Compute tracking covergae
    compute_tracking_coverage(data, args.show);

if __name__ == '__main__':
    main()