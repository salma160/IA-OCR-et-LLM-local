import cv2 as cv
import numpy as np
import easyocr as easy

path_image= r"CIN_photos\WhatsApp Image _2026-07-11 at 20.02.54.jpeg" 
img=cv.imread(path_image)
norm_width=1000
norm_height=630
cv.namedWindow("my pipeline")
cv.createTrackbar("thres1","my pipeline", 75,255,lambda x:x)
cv.createTrackbar("thres2","my pipeline", 200,255,lambda x:x)

def resizefct(img,r=0.75):
    height,width=img.shape[:2]
    width=int(width*r)
    height=int(height*r)
    return cv.resize(img,(width,height))
img=resizefct(img,0.3)

while True:
    th1=cv.getTrackbarPos("thres1","my pipeline")
    th2=cv.getTrackbarPos("thres2","my pipeline")
    img_gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    img_original_gray = img_gray.copy()
    img_blur=cv.GaussianBlur(img_gray,(5,5),0)
    img_canny=cv.Canny(img_blur,th1,th2)
    img_canny=cv.dilate(img_canny,(3,3),1)
    img_canny=cv.erode(img_canny,(3,3),1)
    
    contours, _ = cv.findContours(img_canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
   
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    
    carte_contour = None
    
    for c in contours:
        
        perimetre = cv.arcLength(c, True)
        
        approx = cv.approxPolyDP(c, 0.02 * perimetre, True)
        
        
        if len(approx) == 4:
            carte_contour = approx
            break 
    # --- BLOC DE DIAGNOSTIC ---
    
    
    
    if carte_contour is not None:
        cv.drawContours(img_original_gray, [carte_contour], -1, 255, 3)















    
    
    
    cv.putText(img_original_gray, "Original", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    cv.putText(img_blur, "Gaussian Blur", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    cv.putText(img_canny, "Canny Edge", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    
   
    H_display = cv.hconcat([img_original_gray, img_blur, img_canny])
    cv.imshow("my pipeline",H_display)
    
    if (cv.waitKey(1)&0XFF ) == ord('a'):
        break

cv.destroyAllWindows()
