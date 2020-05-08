import os
import cv2

MAIN_PATH = ''
VIOLENCE = 'violence'
NON_VIOLENCE = 'non_violence'

violent_folders = os.listdir(os.path.join(MAIN_PATH,VIOLENCE))

for violent_folder in violent_folders:
	violent_images = os.listdir(os.path.join(MAIN_PATH,VIOLENCE,violent_folder))
	if len(violent_images) == 0:
		os.rmdir(os.path.join(MAIN_PATH,VIOLENCE,violent_folder))