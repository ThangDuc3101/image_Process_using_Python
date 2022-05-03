import cv2
import numpy as np

# read imgae
img = cv2.imread("digits.png")

# convert to binary image
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# split image
# cells = [np.hsplit(row,100) for row in np.vsplit(imgGray,50)]
cells=[]
for row in np.vsplit(imgGray,50):
    cells.append(np.hsplit(row,100))
# convert splitted images to array
x=np.array(cells)

# device data to training and test
train = x[:,:50].reshape(-1,400).astype(np.float32)
test = x[:,50:100].reshape(-1,400).astype(np.float32)

# create label
k=np.arange(10)
train_label=np.repeat(k,250)[:,np.newaxis]
test_label = train_label.copy()

# create knn
knn = cv2.ml.KNearest_create()
knn.train(train,cv2.ml.ROW_SAMPLE,train_label)
ret,result,neighbourd,dist = knn.findNearest(test,k=5)

# Evaluate
matches = result == test_label
correct = np.count_nonzero(matches)
accuracy = correct*100.0/result.size
print(accuracy)    

# cv2.imshow("Image",img)

if(cv2.waitKey(0)==ord('q')):
    cv2.destroyAllWindow()