#!/bin/bash

# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

set -ex

function usage() {
    echo "Usage: $0 [args]"
    echo ""
    echo "Arguments:"
    echo -e "-v VERSION\t[Required] OpenSearch version."
    echo -e "-s SNAPSHOT\t[Optional] Build a snapshot, default is 'false'."
    echo -e "-p PLATFORM\t[Optional] Platform, ignored."
    echo -e "-a ARCHITECTURE\t[Optional] Build architecture, ignored."
    echo -e "-o OUTPUT\t[Optional] Output path, default is 'artifacts'."
    echo -e "-h help"
}

while getopts ":h:v:s:o:p:a:" arg; do
    case $arg in
        h)
            usage
            exit 1
            ;;
        v)
            VERSION=$OPTARG
            ;;
        s)
            SNAPSHOT=$OPTARG
            ;;
        o)
            OUTPUT=$OPTARG
            ;;
        p)
            PLATFORM=$OPTARG
            ;;
        a)
            ARCHITECTURE=$OPTARG
            ;;
        :)
            echo "Error: -${OPTARG} requires an argument"
            usage
            exit 1
            ;;
        ?)
            echo "Invalid option: -${arg}"
            exit 1
            ;;
    esac
done

if [ -z "$VERSION" ]; then
    echo "Error: You must specify the OpenSearch version"
    usage
    exit 1
fi

[[ "$SNAPSHOT" == "true" ]] && VERSION=$VERSION-SNAPSHOT
[ -z "$OUTPUT" ] && OUTPUT=artifacts

work_dir=$PWD

# Pull library submodule explicitly. While "cmake ." actually pulls the submodule if its not there, we
# need to pull it before calling cmake. Also, we need to call it from the root git directory.
# Otherwise, the submodule update call may fail on earlier versions of git.
git submodule update --init -- jni/external/nmslib

# Build knnlib
cd jni

# For x64, generalize arch so library is compatible for processors without simd instruction extensions
if [ "$ARCHITECTURE" = "x64" ]; then
    sed -i -e 's/-march=native/-march=x86-64/g' external/nmslib/similarity_search/CMakeLists.txt
fi

# For arm, march=native is broken in centos 7. Manually override to lowest version of armv8.
if [ "$ARCHITECTURE" = "arm64" ]; then
    sed -i -e 's/-march=native/-march=armv8-a/g' external/nmslib/similarity_search/CMakeLists.txt
fi

if [ "$JAVA_HOME" = "" ]; then
    export JAVA_HOME=`/usr/libexec/java_home`
    echo "SET JAVA_HOME=$JAVA_HOME"
fi

cmake .
make

cd $work_dir
./gradlew assemble --no-daemon --refresh-dependencies -DskipTests=true -Dopensearch.version=$VERSION -Dbuild.snapshot=$SNAPSHOT

# Add knnlib to zip
zipPath=$(find "$(pwd)" -path \*build/distributions/*.zip)
distributions="$(dirname "${zipPath}")"
mkdir $distributions/knnlib
cp ./jni/release/libKNN* $distributions/knnlib
cd $distributions
zip -ur $zipPath knnlib
cd $work_dir
echo "COPY ${distributions}/*.zip"
mkdir -p $OUTPUT/plugins
cp ${distributions}/*.zip $OUTPUT/plugins
