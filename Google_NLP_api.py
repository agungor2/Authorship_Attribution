# -*- coding: utf-8 -*-
"""
@author: Mecit
"""

from googleapiclient import discovery
import httplib2

service = discovery.build('language', 'v1beta1',
                           http=http, discoveryServiceUrl=DISCOVERY_URL)
                           
service_request = service.documents().analyzeSentiment(
   body={
     'document': {
        'type': 'PLAIN_TEXT',
        'content': "Place your sentence here to do sentiment analysis"
     }
   })

response = service_request.execute()
polarity = response['documentSentiment']['polarity']
magnitude = response['documentSentiment']['magnitude']
print(polarity, magnitude)