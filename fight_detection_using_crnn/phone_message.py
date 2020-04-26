import requests
import json

URL = 'https://www.sms4india.com/api/v1/sendCampaign'

# get request
def sendPostRequest(reqUrl, apiKey, secretKey, useType, phoneNo, senderId, textMessage):
  req_params = {
  'apikey':apiKey,
  'secret':secretKey,
  'usetype':useType,
  'phone': phoneNo,
  'message':textMessage,
  'senderid':senderId
  }
  return requests.post(reqUrl, req_params)

# get response
def main():
    response = sendPostRequest(URL, 'Y0CRJP21JW1CEOE9MNKJ2PD1U2RQKMDC', 'XD326W5W51BT16D4', 'stage', '9925335903', 'Meet', 'message-text' )
"""
  Note:-
    you must provide apikey, secretkey, usetype, mobile, senderid and message values
    and then requst to api
"""
# print response if you want
if __name__ == '__main__':
    print (response.text)

