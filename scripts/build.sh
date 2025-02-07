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
    echo -e "-q QUALIFIER\t[Optional] Version qualifier."
    echo -e "-s SNAPSHOT\t[Optional] Build a snapshot, default is 'false'."
    echo -e "-p PLATFORM\t[Optional] Platform, ignored."
    echo -e "-a ARCHITECTURE\t[Optional] Build architecture, ignored."
    echo -e "-o OUTPUT\t[Optional] Output path, default is 'artifacts'."
    echo -e "-j NPROC_COUNT\t[Optional] Number of CPUs to use when building JNI library. Default is 1."
    echo -e "-h help"
}

while getopts ":h:v:q:s:o:p:a:j:" arg; do
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
        q)
            QUALIFIER=$OPTARG
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
        j)
            NPROC_COUNT=$OPTARG
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

[[ ! -z "$QUALIFIER" ]] && VERSION=$VERSION-$QUALIFIER
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

# For arm, march=native is broken in centos 7. Manually override to lowest version of armv8.
if [ "$ARCHITECTURE" = "arm64" ]; then
    sed -i -e 's/-march=native/-march=armv8-a/g' external/nmslib/similarity_search/CMakeLists.txt
fi

if [ "$JAVA_HOME" = "" ]; then
    export JAVA_HOME=`/usr/libexec/java_home`
    echo "SET JAVA_HOME=$JAVA_HOME"
fi

# Ensure gcc version is above 4.9.0 and at least 9.0.0 for faiss 1.7.4+ / SIMD Neon support on ARM64 compilation
# https://github.com/opensearch-project/k-NN/issues/975
# https://github.com/opensearch-project/k-NN/issues/1138
# https://github.com/opensearch-project/opensearch-build/issues/4386
GCC_VERSION=`gcc --version | head -n 1 | cut -d ' ' -f3`
if [ "$ARCHITECTURE" = "x64" ]; then
  # https://github.com/opensearch-project/opensearch-build/issues/5226
  # We need gcc version >=12.4 to build Faiss Sapphire library(avx512_spr)
  GCC_REQUIRED_VERSION=12.4
else
  GCC_REQUIRED_VERSION=9.0.0
fi
COMPARE_VERSION=`echo $GCC_REQUIRED_VERSION $GCC_VERSION | tr ' ' '\n' | sort -V | uniq | head -n 1`
if [ "$COMPARE_VERSION" != "$GCC_REQUIRED_VERSION" ]; then
    echo "gcc version on this env is older than $GCC_REQUIRED_VERSION, exit 1"
    exit 1
fi

# Build k-NN lib and plugin through gradle tasks
cd $work_dir
./gradlew build --no-daemon --refresh-dependencies -x integTest -x test -Dopensearch.version=$VERSION -Dbuild.snapshot=$SNAPSHOT -Dbuild.version_qualifier=$QUALIFIER -Dbuild.lib.commit_patches=false
./gradlew :buildJniLib -Davx512.enabled=false -Davx512_spr.enabled=false -Davx2.enabled=false -Dbuild.lib.commit_patches=false -Dnproc.count=${NPROC_COUNT:-1}

if [ "$PLATFORM" != "windows" ] && [ "$ARCHITECTURE" = "x64" ]; then
  echo "Building k-NN library nmslib with gcc 10 on non-windows x64"
  rm -rf jni/CMakeCache.txt jni/CMakeFiles
  env CC=gcc10-gcc CXX=gcc10-g++ FC=gcc10-gfortran ./gradlew :buildNmslib -Dbuild.lib.commit_patches=false -Dbuild.lib.apply_patches=false --info

  echo "Building k-NN library after enabling AVX2"
  # Skip applying patches as patches were applied already from previous :buildJniLib task
  # If we apply patches again, it fails with conflict
  rm -rf jni/CMakeCache.txt jni/CMakeFiles
  ./gradlew :buildJniLib -Davx2.enabled=true -Davx512.enabled=false -Davx512_spr.enabled=false -Dbuild.lib.commit_patches=false -Dbuild.lib.apply_patches=false

  echo "Building k-NN library after enabling AVX512"
  ./gradlew :buildJniLib -Davx512.enabled=true -Davx512_spr.enabled=false -Dbuild.lib.commit_patches=false -Dbuild.lib.apply_patches=false

  echo "Building k-NN library after enabling AVX512_SPR"
  ./gradlew :buildJniLib -Davx512_spr.enabled=true -Dbuild.lib.commit_patches=false -Dbuild.lib.apply_patches=false

else
  ./gradlew :buildNmslib -Dbuild.lib.commit_patches=false -Dbuild.lib.apply_patches=false --info
fi

./gradlew publishPluginZipPublicationToZipStagingRepository -Dopensearch.version=$VERSION -Dbuild.snapshot=$SNAPSHOT -Dbuild.version_qualifier=$QUALIFIER
./gradlew publishPluginZipPublicationToMavenLocal -Dbuild.snapshot=$SNAPSHOT -Dbuild.version_qualifier=$QUALIFIER -Dopensearch.version=$VERSION

# Add lib to zip
zipPath=$(find "$(pwd)/build/distributions" -path \*.zip)
distributions="$(dirname "${zipPath}")"
mkdir -p $distributions/lib
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

mkdir -p $OUTPUT/maven/org/opensearch
cp -r ./build/local-staging-repo/org/opensearch/. $OUTPUT/maven/org/opensearch
