import os 
import cv2 


PATH = "Scraped Videos/"
SAVE_PATH = "images/Violence reddit"
currentframe = 0
j = 0

for path in os.listdir(PATH):
    cam = cv2.VideoCapture(os.path.join(PATH,path)) 
    print(path)
      
    while(True): 
        
        ret,frame = cam.read() 
        width  = cam.get(cv2.CAP_PROP_FRAME_WIDTH)  
        height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT) 
        
        
        if ret: 
            if height < width:
                if currentframe % 10 == 0:
                    name = os.path.join(SAVE_PATH, str(j)) + '.jpg'
                    cv2.imwrite(name, frame)
            else:
                cam.release() 
                os.remove(os.path.join(PATH,path))
                break 

            currentframe += 1
            j+=1
        else: 
            break

    print('done')  
    cam.release() 
    cv2.destroyAllWindows() 