import os
import cv2

MAIN_PATH = ''
VIOLENCE = 'violence'
NON_VIOLENCE = 'non_violence'

violent_folders = os.listdir(os.path.join(MAIN_PATH,VIOLENCE))

for violent_folder in violent_folders:
	violent_images = os.listdir(os.path.join(MAIN_PATH,VIOLENCE,violent_folder))
	for violent_image in violent_images:
		image_path= os.path.join(MAIN_PATH,VIOLENCE,violent_folder,violent_image)
		img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) 
		dimensions = img.shape

		print(image_path,"    ",dimensions)

		if not dimensions == (224,224,3):
			os.remove(image_path)
