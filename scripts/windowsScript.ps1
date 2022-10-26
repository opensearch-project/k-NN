#
# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#

git submodule update --init -- jni/external/nmslib
git submodule update --init -- jni/external/faiss

# _MSC_VER is a predefined macro which defines the version of Visual Studio Compiler
# As we are using x86_64-w64-mingw32-gcc compiler we need to replace this macro with __MINGW32__
(Get-Content jni/external/faiss/faiss/impl/index_read.cpp).replace('_MSC_VER', '__MINGW32__') | Set-Content jni/external/faiss/faiss/impl/index_read.cpp
(Get-Content jni/external/faiss/faiss/impl/index_write.cpp).replace('_MSC_VER', '__MINGW32__') | Set-Content jni/external/faiss/faiss/impl/index_write.cpp

# <sys/mman.h> is a Unix header and is not available on Windows. So, adding condition to include it if not running on Windows
# Replace '#include <sys/mman.h>' with
#  #ifndef __MINGW32__
#    #include <sys/mman.h>
#  #endif
(Get-Content jni/external/faiss/faiss/OnDiskInvertedLists.cpp).replace('#include <sys/mman.h>', "#ifndef __MINGW32__`n#include <sys/mman.h>`n#endif") | Set-Content jni/external/faiss/faiss/OnDiskInvertedLists.cpp
