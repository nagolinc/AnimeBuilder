#

import os
import glob
import shutil

#remove tmp.log
if os.path.exists('tmp.log'):
    os.remove('tmp.log')

#remove database if it exists
if os.path.exists('movie_elements.db'):
    os.remove('movie_elements.db')

#remove /pits/logs/
if os.path.exists('./pits/logs/'):
    shutil.rmtree('./pits/logs/')

#remove any file that starts with 'movie_elements.db'    
files=glob.glob("./movie_elements.db*")
for f in files:
    os.remove(f)

#remove any directories in /static/samples, even if they arent' empty
dirs=glob.glob("./static/samples/*/")
for d in dirs:
    shutil.rmtree(d)

#remove any jpgs
jpgs=glob.glob("./static/samples/*.jpg")
for jpg in jpgs:
    os.remove(jpg)
    

#remove any .pickle files in /static/samples
pickles=glob.glob("./static/samples/*.pickle")
for p in pickles:
    os.remove(p)

pngs=glob.glob("./static/samples/*.png")
for png in pngs:
    os.remove(png)

audio=glob.glob("./static/samples/*.mp3")
for f in audio:
    os.remove(f)

audio=glob.glob("./static/samples/*.wav")
for f in audio:
    os.remove(f)    

