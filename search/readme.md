# Setup
1. Download Milvus:
``` bash
cd vector_search

curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh

bash standalone_embed.sh start
```

To stop milvus:
``` bash
bash standalone_embed.sh stop
```

2. Setup ElasticSearch:
``` bash
cd text_search

docker-compose -f docker-compose.elk.yml up -d
```