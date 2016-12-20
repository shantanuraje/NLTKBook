# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 00:52:52 2016

@author: shant
"""

# script to read python history using readline and (todo) output to text file

import readline
for i in range(readline.get_current_history_length()):
    print(readline.get_history_item(i + 1))
