# Finds all files that don't have a matching wave and textgrid

import os

wav = []
text = []

for root, dirs, files in os.walk("/home2/lpanfili/dissertation/Hmong/Hmong_Praat_3/"):  
    for filename in files:
        if filename[-4:] == '.wav':
            wav.append(filename[:-4])
        if filename[-4:] == 'Grid':
            text.append(filename[:-9])

for i in text:
    if i not in wav:
        print(i)