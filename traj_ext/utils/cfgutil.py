# -*- coding: utf-8 -*-

import configparser
import sys

def compute_highest_occurence(item_list):
    """Compute highest occurence from a list

    Args:
        item_list (TYPE): Description

    Returns:
        TYPE: Max item, Occurence
    """
    unique_item_list = remove_duplicates(item_list);

    max_count = 0;
    max_item = None;
    for item in unique_item_list:

        count = item_list.count(item);
        if count > max_count:
            max_count = count;
            max_item = item;

    return max_item, max_count;


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

# https://gist.github.com/vladignatyev/06860ec2040cb497f0f3
def progress_bar(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '#' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()  # As suggested by Rom Ruben (see: http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/27871113#comment50529068_27871113)