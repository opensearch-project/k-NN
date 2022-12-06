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
git submodule update --init -- jni/external/faiss

# Setup compile time dependency for Windows only
# As Linux version already have OpenBlas in the runner
if [ "$PLATFORM" = "windows" ]; then
    openBlasVersion="0.3.21"
    openBlasFile="openblas_${openBlasVersion}"
    curl -SL https://github.com/xianyi/OpenBLAS/releases/download/v${openBlasVersion}/OpenBLAS-${openBlasVersion}-x64.zip -o ${openBlasFile}.zip
    unzip -j -o ${openBlasFile}.zip bin/libopenblas.dll -d ./src/main/resources/windowsDependencies
    rm -rf ${openBlasFile}.zip
fi

# Setup knnlib build params for all platforms
cd jni

# For x64, generalize arch so library is compatible for processors without simd instruction extensions
if [ "$ARCHITECTURE" = "x64" ]; then
    sed -i -e 's/-march=native/-march=x86-64/g' external/nmslib/similarity_search/CMakeLists.txt
fi

# For arm, march=native is broken in centos 7. Manually override to lowest version of armv8. Also, disable simd in faiss
# file. This is broken on centos 7 as well.
if [ "$ARCHITECTURE" = "arm64" ]; then
    sed -i -e 's/-march=native/-march=armv8-a/g' external/nmslib/similarity_search/CMakeLists.txt
    sed -i -e 's/__aarch64__/__undefine_aarch64__/g' external/faiss/faiss/utils/distances_simd.cpp
fi

if [ "$JAVA_HOME" = "" ]; then
    export JAVA_HOME=`/usr/libexec/java_home`
    echo "SET JAVA_HOME=$JAVA_HOME"
fi

# Build k-NN lib and plugin through gradle tasks
cd $work_dir
# Gradle build is used here to replace gradle assemble due to build will also call cmake and make before generating jars
./gradlew build --no-daemon --refresh-dependencies -x integTest -x test -x jacocoTestReport -DskipTests=true -Dopensearch.version=$VERSION -Dbuild.snapshot=$SNAPSHOT -Dbuild.version_qualifier=$QUALIFIER

# Add lib to zip
zipPath=$(find "$(pwd)" -path \*build/distributions/*.zip)
distributions="$(dirname "${zipPath}")"
mkdir $distributions/lib
libPrefix="libopensearchknn"
if [ "$PLATFORM" = "windows" ]; then
    libPrefix="opensearchknn"
    cp -v ./src/main/resources/windowsDependencies/libopenblas.dll $distributions/lib

    # Have to define $MINGW_BIN either in ENV VAR or User Provided Var
    cp -v "$MINGW_BIN/libgcc_s_seh-1.dll" $distributions/lib
    cp -v "$MINGW_BIN/libwinpthread-1.dll" $distributions/lib
    cp -v "$MINGW_BIN/libstdc++-6.dll" $distributions/lib
    cp -v "$MINGW_BIN/libgomp-1.dll" $distributions/lib
else
   ompPath=$(ldconfig -p | grep libgomp | cut -d ' ' -f 4)
   cp -v $ompPath $distributions/lib
fi
cp -v ./jni/release/${libPrefix}* $distributions/lib
ls -l $distributions/lib

# Add lib directory to the k-NN plugin zip
cd $distributions
zip -ur $zipPath lib
cd $work_dir

echo "COPY ${distributions}/*.zip"
mkdir -p $OUTPUT/plugins
cp -v ${distributions}/*.zip $OUTPUT/plugins
