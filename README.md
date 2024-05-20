<h1>Leash-Bio Drug Target Interaction Prediction</h1>

<h2>Setup</h2>

When working on a new machine, run:

```
git clone https://paward35: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@github.com/SeanGormann/leash_dti.git
cd leash_dti/
chmod +x ./setup.sh
./setup.sh
```

<h2>Datasets </h2>

<h3>5 Million</h3>

```
wget 'https://storage.googleapis.com/kaggle-data-sets/5014460/8422528/compressed/30mNegs_allp.csv/30mNegs_allp.csv.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240515%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240515T161332Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=7fbd43b1cd31e38bd242b85ca3645abfa41e61f8aad0be15b45d2b5b681a7bbcc8fcc72f100b7fc797faac32d16e2cff36e993d7f3484632bdbb77f6a30b11b335c8419eb8ea4ac0b28b05e31165303c5a0e1e8b8fac5c8ce5c59ba0fe98f11410da63869419dd38543de558809623becedc04da6f3142b1677a60f5e3c965deff674ed9f4d064290bcfa6d72e15d385cf2e763e24db008165a82389e8e0508d0603ba7edf896413206f2dad561177f33d3fe0ea21f3028b02cb396f4e69422a807b72670a8873ff4e94c225267a3572f1e53e1f140595972c7a9e433375f2bc4f902419e9f810e514accee0c5158e8ed03e1da93e74f464fa63890300f12629' --no-check-certificate -O graph_data_30milly.zip
cd data
unzip graph_data_5milly.zip
cd ..
```

<h3>11 Million</h3>

```
wget 'https://storage.googleapis.com/kaggle-data-sets/4936645/8310370/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240507%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240507T142559Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=1ebfdf77d4a7131d53629f37a9ef06b2785758782096673e29453ad15ecf9d29158d70840ce115aac8d28134d48dc7613fcfe65fa88a7d504a2a84a53cd188db400e68e6c5666164ce38c9bc4d1efc264c8f37485571d9b80aa090074d4e22cc37284fcddddd3506e8f1620f986ad463be0ccfe2729ee1d128f7f6b9a6aa629a10de705a1bc77c0e406ea4f34a3653ba31b211c959471be7e15734abb99e9f5ea0d7dde14197a284a94da6edb7795c9be68d523ba33d2ee704bfb505c5e2a933bb87f87c6a9e5dbacb53433f8c327b4b2144bbaec29c496b1be42893a7689829157623f0bd3a9eccfa91fb44555987c0fc992363d80964d9294eb20b47e98c9c' --no-check-certificate -O graph_dataset_11m.zip
cd data
unzip graph_dataset_11m.zip
cd ..
```

<h3>30 Million</h3>

```
wget 'https://storage.googleapis.com/kaggle-data-sets/5014460/8422528/compressed/5mNegs_allp.csv.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240515%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240515T161116Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=2d71f12444712c64a7d420bd709eb5ef6750d7a7011b30a7ae04b4171444354311481d87182ad315b779a1a722622a6c0b8f85d457504ae1342e62cda46320c9bc1ac068321a6b27f0ad8901e966c6c2ebd39501aa6ba719a6ec42eae2d32395fb108945b5bd44d522761672c650aa922f33325a11128073a49eda7af7ad1cc5a2cd17681fa61f7d45e9e40732303ba358578d377096549493922ad6e36a59dceaee7cdb0720fb977ae077e255601c340341909768e31b532f0c4fbbd840dbdc8dca238d4465d8d37f02ba058d85c5f07a6d89e1fc17e3ad925def750fec9e2008fd9966e783a8dd9ad5da459dbb66193970903f6e0710ee56556c54640fe881' --no-check-certificate -O graph_data_5milly.zip
cd data
unzip graph_data_30milly.zip
cd ..
```

<h2>learning </h2>

```
python main.py
```
