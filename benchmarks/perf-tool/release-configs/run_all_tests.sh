#!/bin/bash
set -e

# Description:
# Run a performance test for release
# Dataset should be available in perf-tool/dataset before running this script
#
# Example:
# ./run-test.sh --endpoint localhost
#
# Usage:
# ./run-test.sh \
#   --endpoint <your endpoint>
#   --port 80 \
#   --num-runs 3 \
#   --outputs ~/outputs

while [ "$1" != "" ]; do
  case $1 in
    -url | --endpoint )    shift
                        ENDPOINT=$1
                        ;;
    -p | --port )    shift
                        PORT=$1
                        ;;
    -n | --num-runs )    shift
                        NUM_RUNS=$1
                        ;;
    -o | --outputs )    shift
                        OUTPUTS=$1
                        ;;
    * )                 echo "Unknown parameter"
                        echo $1
                        exit 1
                        ;;
  esac
  shift
done

if [ ! -n "$ENDPOINT" ]; then
    echo "--endpoint should be specified"
    exit
fi

if [ ! -n "$PORT" ]; then
        PORT=80
        echo "--port is not specified. Using default values $PORT"
fi

if [ ! -n "$NUM_RUNS" ]; then
        NUM_RUNS=3
        echo "--num-runs is not specified. Using default values $NUM_RUNS"
fi

if [ ! -n "$OUTPUTS" ]; then
        OUTPUTS="$HOME/outputs"
        echo "--outputs is not specified. Using default values $OUTPUTS"
fi


curl -X PUT "http://$ENDPOINT:$PORT/_cluster/settings?pretty" -H 'Content-Type: application/json' -d'
{
 "persistent" : {
   "knn.algo_param.index_thread_qty" : 4
 }
}
'

TESTS="./release-configs/faiss-hnsw/filtering/relaxed-filter/relaxed-filter-test.yml
./release-configs/faiss-hnsw/filtering/restrictive-filter/restrictive-filter-test.yml
./release-configs/faiss-hnsw/nested/simple/simple-nested-test.yml
./release-configs/faiss-hnsw/test.yml
./release-configs/faiss-hnswpq/test.yml
./release-configs/faiss-hnsw-sqfp16/test.yml
./release-configs/faiss-ivf/filtering/relaxed-filter/relaxed-filter-test.yml
./release-configs/faiss-ivf/filtering/restrictive-filter/restrictive-filter-test.yml
./release-configs/faiss-ivf/test.yml
./release-configs/faiss-ivfpq/test.yml
./release-configs/faiss-ivf-sqfp16/test.yml
./release-configs/lucene-hnsw/filtering/relaxed-filter/relaxed-filter-test.yml
./release-configs/lucene-hnsw/filtering/restrictive-filter/restrictive-filter-test.yml
./release-configs/lucene-hnsw/nested/simple/simple-nested-test.yml
./release-configs/lucene-hnsw/test.yml
./release-configs/nmslib-hnsw/test.yml"

if [ ! -d $OUTPUTS ]
then
        mkdir $OUTPUTS
fi

for TEST in $TESTS
do
        ORG_FILE=$TEST
        NEW_FILE="$ORG_FILE.tmp"
        OUT_FILE=$(grep test_id $ORG_FILE | cut -d':' -f2 | sed -r 's/^ "|"$//g' | sed 's/ /_/g')
        echo "cp $ORG_FILE $NEW_FILE"
        cp $ORG_FILE $NEW_FILE
        sed -i "/^endpoint:/c\endpoint: $ENDPOINT" $NEW_FILE
        sed -i "/^port:/c\port: $PORT" $NEW_FILE
        sed -i "/^num_runs:/c\num_runs: $NUM_RUNS" $NEW_FILE
        python3 knn-perf-tool.py test $NEW_FILE $OUTPUTS/$OUT_FILE
        #Sleep for 1 min to cool down cpu from the previous run
        sleep 60
done
