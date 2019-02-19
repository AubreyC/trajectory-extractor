# -*- coding: utf-8 -*-

import configparser

def read_cfg(config_path):

    if config_path == '':
        raise ValueError('[Error]: Config file path is empty');

    # Read config file:
    config = configparser.ConfigParser();
    config.read(config_path);

    if config.sections() == []:
        raise ValueError('[Error]: Config file is empty: {}'.format(config_path));

    return config;

# https://thispointer.com/python-how-to-remove-duplicates-from-a-list/
def remove_duplicates(listofElements):

    # Create an empty list to store unique elements
    uniqueList = []

    # Iterate over the original list and for each element
    # add it to uniqueList, if its not already there.
    for elem in listofElements:
        if elem not in uniqueList:
            uniqueList.append(elem)

    # Return the list of unique elements
    return uniqueList

def compute_highest_occurence(item_list):

    unique_item_list = remove_duplicates(item_list);

    max_count = 0;
    max_item = None;
    for item in unique_item_list:

        count = item_list.count(item);
        if count > max_count:
            max_count = count;
            max_item = item;

    return max_item;