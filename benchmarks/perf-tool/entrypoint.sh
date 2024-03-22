#!/bin/bash

set -ex


cd /home/opensearch
conda create -n knn-perf python=3.8 -y --force
source activate knn-perf
pip install -r requirements.txt
#curl -v http://os-node1:9200/
cd /home/opensearch
python knn-perf-tool.py --log info test /PATH/PATH_TO.yml output.json
