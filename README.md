# byollm
Build your own LLM


**Transformers**

Run the following commands to download IMDB Sentiment dataset

`
mkdir -p ./data/sentiment
wget -P ./data/sentiment https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xf ./data/sentiment/aclImdb_v1.tar.gz -C ./data/sentiment
rm -rf ./data/sentiment/aclImdb/train/unsup
`

create **models** folder to store models artifacts
`
mkdir ./models
`
