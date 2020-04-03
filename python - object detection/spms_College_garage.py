# -*- coding: utf-8 -*-
"""

"""


import cv2
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt
import glob

# %config InlineBackend.figure_format = 'svg'

# define the model options and run

options = {
     'model': 'cfg/tiny-yolo-voc-1c.cfg',
    'load': 20000,
    'threshold': 0.1,
#   'gpu': 1.0
}

tfnet = TFNet(options)

# read the color image and covert to RGB
############## College Parking id #################

ParkingId = 17          # Tird Id     ## 11 parking space

############################################
def update(filled,lot):
    import urllib.request
    link = "http://spms.msa3d.com/api.php?get=python&filled="+str(filled)+"&lot="+str(lot)
    contents = urllib.request.urlopen(link).read()

    
###########


def videoDetection(frame,id):
 
    result = tfnet.return_predict(frame)
    for i in range(len(result)):
        tl = (result[i]['topleft']['x'], result[i]['topleft']['y'])
        br = (result[i]['bottomright']['x'], result[i]['bottomright']['y'])
        label = result[i]['label']
        confidence = result[i]['confidence']
        text = '{}:{:.0f}%'.format(label, confidence * 100)
        frame = cv2.rectangle(frame, tl, br, (0, 255, 0), 7)
        frame = cv2.putText(frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
    ID=str(id)      # convert integer to string
    frame = cv2.resize(frame, (960, 540))
    cv2.imshow(ID, frame)
    update(len(result),id)
    return len(result)

tfnet = TFNet(options)

# read the color image and covert to RGB
images = [cv2.imread(file) for file in glob.glob("video/*.jpg")]
for img in images :  



    car_counter= videoDetection(img,ParkingId)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Parking Cars = ", car_counter)

    #time.sleep(0.5)   ## Edit time delay for it

    # When everything done, release the capture
    

# plt.imshow(img)
# plt.show()
