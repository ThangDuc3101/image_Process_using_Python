from cv2 import cv2

# đọc hình ảnh
img = cv2.imread("digits.png")

# chuyển xám 
imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# chia nhỏ
import numpy as np
cells=[]
for row in np.vsplit(imgGray,50):
    cells.append(np.hsplit(row,100))

# chuyển từng ảnh thành ma trận số        
x=np.array(cells)

# chia dũ liệu + duỗi ảnh
train = x[:,:50].reshape(-1,400).astype(np.float32)
test = x[:,50:100].reshape(-1,400).astype(np.float32)

# tạo nhãn
k = np.arange(10)
train_labels=np.repeat(k,250)[:,np.newaxis]
test_labels=train_labels.copy()

# tạo moduel knn
knn = cv2.ml.KNearest_create()
knn.train(train,cv2.ml.ROW_SAMPLE,train_labels)
results = knn.findNearest(test, k=5)

# đánh giá

matches = results[1]==test_labels
correct = np.count_nonzero(matches)
accuracy = correct*100.0/(results[1].size)
print(accuracy)