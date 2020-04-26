import requests
import json

URL = 'https://www.sms4india.com/api/v1/sendCampaign'

# get request
def sendPostRequest( apiKey, secretKey, phoneNo, senderId, textMessage,reqUrl= URL,useType='stage'):
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
response = sendPostRequest('Y0CRJP21JW1CEOE9MNKJ2PD1U2RQKMDC', 'XD326W5W51BT16D4', '9265983776', 'Parth', 'message-text' )
"""
  Note:-
    you must provide apikey, secretkey, usetype, mobile, senderid and message values
    and then requst to api
"""
