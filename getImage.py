#!/usr/bin/python
import json
from pymongo import MongoClient
clientFrom = MongoClient("mongodb://<username>:<password>@ds059145.mlab.com:59145/tweets") # Teacher's database
dbFrom = clientFrom.tweets

clientTo = MongoClient("mongodb://<username>:<password>@ds145355.mlab.com:45355/justurl") # Our db to store just Url
dbTo = clientTo.justurl

cursor = dbFrom.JsonResponse.find() # get all tweets from teacher's database

for document in cursor:
   # get just JsonResponse object from Db
    try:
        text = document['user']['profile_image_url']
        # text2 = document['entities']['media'][0]['media_url']
    except:
        continue
    print text
    result = dbTo.Url.insert_one({ "image" : text}) 

