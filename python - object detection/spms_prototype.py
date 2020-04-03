# -*- coding: utf-8 -*-
"""

"""


import cv2
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt
import time

# %config InlineBackend.figure_format = 'svg'

# define the model options and run

options = {
    'model': 'cfg/tiny-yolo-voc-1c.cfg',
    'load': 4480,
    'threshold': 0.1,
   'gpu': 1.0
}
############ IDs ############
AswanEngineeringId = 17     # First Id
HelnanId = 18               #Second Id
############
def update(filled,lot):
    import urllib.request
    link = "http://spms.msa3d.com/api.php?get=python&filled="+str(filled)+"&lot="+str(lot)
    contents = urllib.request.urlopen(link).read()

    
###########


def videoDetection(cap,id):
    ret, frame = cap.read()
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
    cv2.imshow(ID, frame)
    update(len(result),id)
    return len(result)

tfnet = TFNet(options)

# read the color image and covert to RGB

cap = cv2.VideoCapture(2)   #id=1
cap2= cv2.VideoCapture(1)   #id=2
while(True):

    car_counter = videoDetection(cap,AswanEngineeringId)
    car_counter2= videoDetection(cap2,HelnanId)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    print("Parking 1 = ", car_counter)
    print("Parking 2 = ", car_counter2)

    time.sleep(0.5)   ## Edit time delay for it

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

# plt.imshow(img)
# plt.show()
