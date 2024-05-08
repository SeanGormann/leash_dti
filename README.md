# leash_dti
Leash-Bio Drug Target Interaction Prediction 


## Ayooooo

Paddy after downloading this go to your terminal and cd to the data folder and then run this command:

***

wget 'https://storage.googleapis.com/kaggle-data-sets/4936645/8310370/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240507%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240507T142559Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=1ebfdf77d4a7131d53629f37a9ef06b2785758782096673e29453ad15ecf9d29158d70840ce115aac8d28134d48dc7613fcfe65fa88a7d504a2a84a53cd188db400e68e6c5666164ce38c9bc4d1efc264c8f37485571d9b80aa090074d4e22cc37284fcddddd3506e8f1620f986ad463be0ccfe2729ee1d128f7f6b9a6aa629a10de705a1bc77c0e406ea4f34a3653ba31b211c959471be7e15734abb99e9f5ea0d7dde14197a284a94da6edb7795c9be68d523ba33d2ee704bfb505c5e2a933bb87f87c6a9e5dbacb53433f8c327b4b2144bbaec29c496b1be42893a7689829157623f0bd3a9eccfa91fb44555987c0fc992363d80964d9294eb20b47e98c9c' --no-check-certificate -O graph_dataset_11m.zip

***

It'll download a zip file with the 11million samples (1.5 positives and the rest negatives). There should be about 30 batches of around 360k graphs in each batch. Unzip the data.

After, 'cd ..' to the main folder, create a virtual env and run pip install -r requirements.txt 

You should be able to test it out instantly after that, but there's a chance you might need to change some of the paths around. Test by running 'python main.py' anyway and see what happens. 
