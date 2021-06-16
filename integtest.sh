#!/bin/bash

set -e

function usage() {
    echo ""
    echo "This script is used to run integration tests for plugin installed on a remote OpenSearch/Dashboards cluster."
    echo "--------------------------------------------------------------------------"
    echo "Usage: $0 [args]"
    echo ""
    echo "Required arguments:"
    echo "None"
    echo ""
    echo "Optional arguments:"
    echo -e "-b BIND_ADDRESS\t, defaults to localhost | 127.0.0.1, can be changed to any IP or domain name for the cluster location."
    echo -e "-p BIND_PORT\t, defaults to 9200 or 5601 depends on OpenSearch or Dashboards, can be changed to any port for the cluster location."
    echo -e "-s SECURITY_ENABLED\t(true | false), defaults to true. Specify the OpenSearch/Dashboards have security enabled or not."
    echo -e "-c CREDENTIAL\t(usename:password), no defaults, effective when SECURITY_ENABLED=true."
    echo -e "-h\tPrint this message."
    echo "--------------------------------------------------------------------------"
}

while getopts ":hb:p:s:c:" arg; do
    case $arg in
        h)
            usage
            exit 1
            ;;
        b)
            BIND_ADDRESS=$OPTARG
            ;;
        p)
            BIND_PORT=$OPTARG
            ;;
        s)
            SECURITY_ENABLED=$OPTARG
            ;;
        c)
            CREDENTIAL=$OPTARG
            ;;
        :)
            echo "-${OPTARG} requires an argument"
            usage
            exit 1
            ;;
        ?)
            echo "Invalid option: -${OPTARG}"
            exit 1
            ;;
    esac
done


if [ -z "$BIND_ADDRESS" ]
then
  BIND_ADDRESS="localhost"
fi

if [ -z "$BIND_PORT" ]
then
  BIND_PORT="9200"
fi

if [ -z "$SECURITY_ENABLED" ]
then
  SECURITY_ENABLED="true"
fi

if [ -z "$CREDENTIAL" ]
then
  CREDENTIAL="admin:admin"
  USERNAME=`echo $CREDENTIAL | awk -F ':' '{print $1}'`
  PASSWORD=`echo $CREDENTIAL | awk -F ':' '{print $2}'`
fi

./gradlew integTest -Dtests.rest.cluster="$BIND_ADDRESS:$BIND_PORT" -Dtests.cluster="$BIND_ADDRESS:$BIND_PORT" -Dtests.clustername="opensearch-integrationtest" -Dhttps=$SECURITY_ENABLED -Duser=$USERNAME -Dpassword=$PASSWORD --console=plain

