/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

/* JNI declarations for org.opensearch.knn.sandbox.svs.SvsService */

#include <jni.h>

#ifndef _Included_org_opensearch_knn_sandbox_svs_SvsService
#define _Included_org_opensearch_knn_sandbox_svs_SvsService
#ifdef __cplusplus
extern "C" {
#endif

/*
 * Class:     org_opensearch_knn_sandbox_svs_SvsService
 * Method:    isLvqLeanvecEnabled
 * Signature: ()Z
 */
JNIEXPORT jboolean JNICALL Java_org_opensearch_knn_sandbox_svs_SvsService_isLvqLeanvecEnabled
  (JNIEnv *, jclass);

/*
 * Class:     org_opensearch_knn_sandbox_svs_SvsService
 * Method:    initIndex
 * Signature: (JILjava/util/Map;)J
 */
JNIEXPORT jlong JNICALL Java_org_opensearch_knn_sandbox_svs_SvsService_initIndex
  (JNIEnv *, jclass, jlong, jint, jobject);

/*
 * Class:     org_opensearch_knn_sandbox_svs_SvsService
 * Method:    insertToIndex
 * Signature: ([IJIJI)V
 */
JNIEXPORT void JNICALL Java_org_opensearch_knn_sandbox_svs_SvsService_insertToIndex
  (JNIEnv *, jclass, jintArray, jlong, jint, jlong, jint);

/*
 * Class:     org_opensearch_knn_sandbox_svs_SvsService
 * Method:    writeIndex
 * Signature: (JLorg/opensearch/knn/index/store/IndexOutputWithBuffer;)V
 */
JNIEXPORT void JNICALL Java_org_opensearch_knn_sandbox_svs_SvsService_writeIndex
  (JNIEnv *, jclass, jlong, jobject);

/*
 * Class:     org_opensearch_knn_sandbox_svs_SvsService
 * Method:    loadIndexWithStream
 * Signature: (Lorg/opensearch/knn/index/store/IndexInputWithBuffer;)J
 */
JNIEXPORT jlong JNICALL Java_org_opensearch_knn_sandbox_svs_SvsService_loadIndexWithStream
  (JNIEnv *, jclass, jobject);

/*
 * Class:     org_opensearch_knn_sandbox_svs_SvsService
 * Method:    queryIndex
 * Signature: (J[FILjava/util/Map;)[Lorg/opensearch/knn/index/query/KNNQueryResult;
 */
JNIEXPORT jobjectArray JNICALL Java_org_opensearch_knn_sandbox_svs_SvsService_queryIndex
  (JNIEnv *, jclass, jlong, jfloatArray, jint, jobject);

/*
 * Class:     org_opensearch_knn_sandbox_svs_SvsService
 * Method:    queryIndexWithFilter
 * Signature: (J[FILjava/util/Map;[JI)[Lorg/opensearch/knn/index/query/KNNQueryResult;
 */
JNIEXPORT jobjectArray JNICALL Java_org_opensearch_knn_sandbox_svs_SvsService_queryIndexWithFilter
  (JNIEnv *, jclass, jlong, jfloatArray, jint, jobject, jlongArray, jint);

/*
 * Class:     org_opensearch_knn_sandbox_svs_SvsService
 * Method:    free
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_org_opensearch_knn_sandbox_svs_SvsService_free
  (JNIEnv *, jclass, jlong);

/*
 * Class:     org_opensearch_knn_sandbox_svs_SvsService
 * Method:    initLibrary
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_org_opensearch_knn_sandbox_svs_SvsService_initLibrary
  (JNIEnv *, jclass);

/*
 * Class:     org_opensearch_knn_sandbox_svs_SvsService
 * Method:    setMergeInterruptCallback
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_org_opensearch_knn_sandbox_svs_SvsService_setMergeInterruptCallback
  (JNIEnv *, jclass);

#ifdef __cplusplus
}
#endif
#endif
