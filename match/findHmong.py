# Finds all files that don't have a matching wave and textgrid

import os
import glob

for root, dirs, files in os.walk("/Users/Laura/Desktop/"): 
    for filename in files:
        print(filename)