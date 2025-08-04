#
# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#

git submodule update --init -- jni/external/nmslib
git submodule update --init -- jni/external/faiss

# _MSC_VER is a predefined macro which defines the version of Visual Studio Compiler
# As we are using x86_64-w64-mingw32-gcc compiler we need to replace this macro with __MINGW64__
(Get-Content jni/external/faiss/faiss/impl/index_read.cpp).replace('_MSC_VER', '__MINGW64__') | Set-Content jni/external/faiss/faiss/impl/index_read.cpp
(Get-Content jni/external/faiss/faiss/impl/index_write.cpp).replace('_MSC_VER', '__MINGW64__') | Set-Content jni/external/faiss/faiss/impl/index_write.cpp
(Get-Content jni/external/faiss/faiss/impl/platform_macros.h).replace('_MSC_VER', '__MINGW64__') | Set-Content jni/external/faiss/faiss/impl/platform_macros.h
(Get-Content jni/external/faiss/faiss/impl/platform_macros.h).replace('#define __PRETTY_FUNCTION__ __FUNCSIG__', ' ') | Set-Content jni/external/faiss/faiss/impl/platform_macros.h
(Get-Content jni/external/faiss/faiss/utils/utils.cpp).replace('_MSC_VER', '__MINGW64__') | Set-Content jni/external/faiss/faiss/utils/utils.cpp
(Get-Content jni/external/faiss/faiss/utils/prefetch.h).replace('_MSC_VER', '__MINGW64__') | Set-Content jni/external/faiss/faiss/utils/prefetch.h
(Get-Content jni/external/faiss/faiss/invlists/InvertedListsIOHook.cpp).replace('_MSC_VER', '__MINGW64__') | Set-Content jni/external/faiss/faiss/invlists/InvertedListsIOHook.cpp
(Get-Content jni/external/faiss/faiss/AutoTune.cpp).replace('__PRETTY_FUNCTION__', 'NULL') | Set-Content jni/external/faiss/faiss/AutoTune.cpp
(Get-Content jni/external/faiss/faiss/utils/distances_simd.cpp).replace('FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN', ' ') | Set-Content jni/external/faiss/faiss/utils/distances_simd.cpp
(Get-Content jni/external/faiss/faiss/utils/distances_simd.cpp).replace('FAISS_PRAGMA_IMPRECISE_FUNCTION_END', ' ') | Set-Content jni/external/faiss/faiss/utils/distances_simd.cpp



# <sys/mman.h> is a Unix header and is not available on Windows. So, adding condition to include it if not running on Windows
# Replace '#include <sys/mman.h>' with
#  #ifndef __MINGW64__
#    #include <sys/mman.h>
#  #endif
(Get-Content jni/external/faiss/faiss/invlists/OnDiskInvertedLists.cpp).replace('#include <sys/mman.h>', "#ifndef __MINGW64__`n#include <sys/mman.h>`n#endif") | Set-Content jni/external/faiss/faiss/invlists/OnDiskInvertedLists.cpp
# intrin.h function like __builtin_ctz, __builtin_clzll is not available in MINGW64. So, adding condition to include it if not running on Windows
# Replace '#include <intrin.h>' with
#  #ifndef __MINGW64__
#    include <intrin.h>
# and
# Closing the above #ifndef with
# #define __builtin_popcountl __popcnt64
# #endif
(Get-Content jni/external/faiss/faiss/impl/platform_macros.h).replace('#include <intrin.h>', "#ifndef __MINGW64__`n#include <intrin.h>`n") | Set-Content jni/external/faiss/faiss/impl/platform_macros.h
(Get-Content jni/external/faiss/faiss/impl/platform_macros.h).replace('#define __builtin_popcountll __popcnt64', "#define __builtin_popcountll __popcnt64`n#endif`n") | Set-Content jni/external/faiss/faiss/impl/platform_macros.h
