#!/usr/bin/python
import json
from pymongo import MongoClient
clientFrom = MongoClient("mongodb://<dbuser>:<dbpassword>@ds031845.mlab.com:31845/backup_20160701") # Teacher's database
dbFrom = clientFrom.backup_20160701

clientTo = MongoClient("mongodb://<dbuser>:<dbpassword>@ds059145.mlab.com:59145/tweets") # Our db to get just JsonResponse object
dbTo = clientTo.tweets

cursor = dbFrom.Response.find() # get all tweets from teacher's database

for document in cursor:
    text = document['JsonResponse'] # get just JsonResponse object from Db
    result = dbTo.JsonResponse.insert_one(json.loads(text)) # add new database JsonResponse object.

