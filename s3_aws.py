import boto3
import os
import io
import pickle as pkl
from pprint import pprint


s3 = boto3.client('s3')

bucket_name = 'tttbucket82'
object_key = 'test.pkl'

''' 
s3 client upload
    with open('data/memory_2187.pkl', 'rb') as file :
        s3.upload_fileobj(file, bucket_name, object_key)
'''

''' s3 client download '''
'''
with open('FILE_NAME', 'wb') as f:
    s3.download_fileobj('BUCKET_NAME', 'OBJECT_NAME', f)
'''

'''s3 client see buckets'''
response = s3.list_buckets()

for bucket in response['Buckets']:
    bucket_name = bucket['Name']
    response = s3.list_objects(Bucket = bucket_name)
    #pprint(response.get('Contents'))

''' s3 resource '''

s3 = boto3.resource('s3')
p_file = s3.Object(bucket_name, object_key)
file_stream = io.BytesIO()
p_file.download_fileobj(file_stream)
file_stream.seek(0)
upo = pkl.load(file_stream)
print(type(upo))

