import os
import re
import praw
import pickle
import requests 
import pandas as pd
import datetime as dt
from urllib.request import urlretrieve

with open('credentials.txt') as f:
	PERSONAL_USE_SCRIPT_14_CHARS = f.readline().rstrip("\n")
	SECRET_KEY_27_CHARS          = f.readline().rstrip("\n")
	YOUR_APP_NAME                = f.readline().rstrip("\n")
	YOUR_REDDIT_USER_NAME        = f.readline().rstrip("\n")
	YOUR_REDDIT_LOGIN_PASSWORD   = f.readline().rstrip("\n")
	SUBREDDIT                    = f.readline().rstrip("\n")


reddit = praw.Reddit(client_id=PERSONAL_USE_SCRIPT_14_CHARS,client_secret=SECRET_KEY_27_CHARS, 
                     user_agent=YOUR_APP_NAME,username=YOUR_REDDIT_USER_NAME, 
                     password=YOUR_REDDIT_LOGIN_PASSWORD)

subreddit = reddit.subreddit(SUBREDDIT)

posts = subreddit.top(limit=10000)

vids = []
names = []

if os.path.exists('found_urls'):
	with open('found_urls', 'rb') as f:
		vids = pickle.load(f)
else:
	with open('found_urls','wb+') as f:
		pickle.dump(vids,f)

if os.path.exists('names'):
	with open('names', 'rb') as f:
		names = pickle.load(f)
else:
	with open('names','wb+') as f:
		pickle.dump(names,f)


print('finding URLs')
quality = ['DASH_240','DASH_360','DASH_480']

for post in posts:
	try:		
		url = post.media['reddit_video']['fallback_url']
		url = url.split("?")[0]
		name = post.title[:30].rstrip() + ".mp4"
		name = re.sub('[^A-Z .a-z0-9]+', '', name)
		if url.split('/')[-1] in quality:
			if url not in vids:	
				vids.append(url)
				names.append(name)
	except:
		pass

with open('found_urls','wb+') as f:
	pickle.dump(vids,f)

with open('names','wb+') as f:
	pickle.dump(names,f)

print('URLs found and stored in file.')

PATH = "Scraped Videos/"

for i in range(len(vids)):
	image_url = vids[i]
	r = requests.get(image_url) 

	if not os.path.exists(PATH + names[i]):
		with open(PATH + names[i],'wb') as video: 
				video.write(r.content)
				print(names[i])

