#Thêm thư viện
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy
from cv2 import cv2

#hiển thị các chức năng có thể lựa chọn
def menu():
    print("===== Chọns chức năng =====")
    print("1.Đọc hiển thị ảnh")
    print("2.Chuyển ảnh sang xám")
    print("3.Hiển thị Histogram")
    print("9.Không gian màu RGB,HSV,CMYK")
    print("10.Nén và hiển thị ảnh")
    print("0.Thoát")
    print("===========================")

menu()  
#chọn chức năng 
choice = int(input())


#thực thi chức năng
while choice!=0:
    if(choice == 0 ):
        break
    elif(choice == 1):
        #yêu cầu 1: đọc ảnh
        #đọc ảnh từ thư mục resource
        img = cv2.imread("resources/panda.jpg")
        #hiển thị ảnh
        cv2.imshow("Image",img)
        # chờ nhấn 1 phím bất kỳ trên bàn phím. Nếu phím đó là 'q' thì xóa hết các cửa sổ hiện có
        if(cv2.waitKey(0)==ord('q')):
            destroyAllWindow()
            break
            
    elif(choice == 2):
        #yêu cầu 2: chuyển sang ảnh xám
        #đọc ảnh từ đường dẫn resource/panda.jpg
        img = cv2.imread("resources/panda.jpg")
        #chuyển bức ảnh vừa đọc dduocj sang thang màu xám
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #hiển thị bức ảnh lên màn hình
        cv2.imshow("Image",img)
        # chờ nhấn 1 phím bất kỳ trên bàn phím. Nếu phím đó là 'q' thì xóa hết các cửa sổ hiện có
        if(cv2.waitKey(0)==ord('q')):
            destroyAllWindow()
            break

    elif(choice == 3):
        #yêu cầu 3: Hiển thị Histogram
        #đọc ảnh từ đường dẫn resource/panda.jpg
        img = cv2.imread('resources/panda.jpg',0)
        #hiện thị bức ảnh gốc lên màn hình
        cv2.imshow("Image",img)
        #hiển thị đồ thị histogram của bức ảnh
        plt.hist(img.ravel(),256,[0,256]); plt.show()
        # chờ nhấn 1 phím bất kỳ trên bàn phím. Nếu phím đó là 'q' thì xóa hết các cửa sổ hiện có
        if(cv2.waitKey(0)==ord('q')):
            destroyAllWindow()
            break

    elif(choice==9):
        #yêu cầu 9: không gian màu rgb,hsv,cmyk
        img = cv2.imread("resources/panda.jpg")
        #chuyển ảnh sang hsv
        imgHSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        #chuyển ảnh sang rgbong
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        #chuyển ảnh sang cmyk
        imgCMYK=cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
        #hiển thị ảnh HSV
        cv2.imshow("ImageHSV",imgHSV)
        #hiển thị ảnh RGB
        cv2.imshow("ImageRGB",imgRGB)
        #hiển thị ảnh CMYK
        cv2.imshow("ImageCMYK",imgCMYK)

        #hiển thị ảnh gốc
        cv2.imshow("Image",img)
        # chờ nhấn 1 phím bất kỳ trên bàn phím. Nếu phím đó là 'q' thì xóa hết các cửa sổ hiện có    
        if(cv2.waitKey(0)==ord('q')):
            destroyAllWindow()
        break
    elif(choice ==10):
        #đọc ảnh từ đường dẫn resources/panda.jpg
        img = plt.imread("resources/panda.jpg")

        image = cv2.imread("resources/panda.jpg")
        #Lấy ra chiều dài và chiều rộng của ảnh gốc
        width = img.shape[0]
        height = img.shape[1]
        #đổi chiều của bức ảnh
        img = img.reshape(width*height,3)
        #Phân cụm các màu thành 5 cụm màu
        kmeans = KMeans(n_clusters=5).fit(img)
        #Tạo ảnh rỗng để chứa bức ảnh sau khi nén
        img2 = numpy.zeros_like(img)
        #ghi dữ liệu lên bứcs ảnh trống 

        labels = kmeans.predict(img)
        clusters = kmeans.cluster_centers_

        for i in range(len(img2)):
            img2[i] = clusters[labels[i]]

        #đưa bức ảnh về định dạng chuẩn
        img2 = img2.reshape(width,height,3)
        #Hiển thị ảnh gốc ban đầu
        cv2.imshow("Image",image)
        #hiển thị ảnh được nén
        plt.imshow(img2)
        plt.show()
        # chờ nhấn 1 phím bất kỳ trên bàn phím. Nếu phím đó là 'q' thì xóa hết các cửa sổ hiện có
        if(cv2.waitKey(0)==ord('q')):
            destroyAllWindow()
        break
    else:
        print("Chức năng vừa lựa chọn không có, vui long chon lại")
        menu()
        choice=int(input())        