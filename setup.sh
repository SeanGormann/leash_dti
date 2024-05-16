echo "cloning repo..."

git clone https://paward35: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@github.com/SeanGormann/leash_dti.git

echo "getting data"
cd leash_dti/data
wget 'https://storage.googleapis.com/kaggle-data-sets/5014460/8422528/compressed/5mNegs_allp.csv.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240515%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240515T161116Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=2d71f12444712c64a7d420bd709eb5ef6750d7a7011b30a7ae04b4171444354311481d87182ad315b779a1a722622a6c0b8f85d457504ae1342e62cda46320c9bc1ac068321a6b27f0ad8901e966c6c2ebd39501aa6ba719a6ec42eae2d32395fb108945b5bd44d522761672c650aa922f33325a11128073a49eda7af7ad1cc5a2cd17681fa61f7d45e9e40732303ba358578d377096549493922ad6e36a59dceaee7cdb0720fb977ae077e255601c340341909768e31b532f0c4fbbd840dbdc8dca238d4465d8d37f02ba058d85c5f07a6d89e1fc17e3ad925def750fec9e2008fd9966e783a8dd9ad5da459dbb66193970903f6e0710ee56556c54640fe881' --no-check-certificate -O graph_data_5milly.zip
apt-get install unzip
unzip graph_data_5milly.zip
cd ..

echo "logging into ECR..."
sudo su – 
$(aws ecr get-login --region us-east-1 --no-include-email --registry-ids 763104351884)

echo "installing requirements"
pip install -r requirements.txt
