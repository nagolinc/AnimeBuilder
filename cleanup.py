#

import os
import glob

pngs=glob.glob("./static/samples/*.png")
for png in pngs:
    os.remove(png)

audio=glob.glob("./static/samples/*.mp3")
for f in audio:
    os.remove(f)

audio=glob.glob("./static/samples/*.wav")
for f in audio:
    os.remove(f)    

