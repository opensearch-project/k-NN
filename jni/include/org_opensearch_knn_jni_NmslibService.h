/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class org_opensearch_knn_jni_NmslibService */

#ifndef _Included_org_opensearch_knn_jni_NmslibService
#define _Included_org_opensearch_knn_jni_NmslibService
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     org_opensearch_knn_jni_NmslibService
 * Method:    createIndex
 * Signature: ([I[[FLjava/lang/String;Ljava/util/Map;)V
 */
JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_NmslibService_createIndex
  (JNIEnv *, jclass, jintArray, jobjectArray, jstring, jobject);

/*
 * Class:     org_opensearch_knn_jni_NmslibService
 * Method:    loadIndex
 * Signature: (Ljava/lang/String;Ljava/util/Map;)J
 */
JNIEXPORT jlong JNICALL Java_org_opensearch_knn_jni_NmslibService_loadIndex
  (JNIEnv *, jclass, jstring, jobject);

/*
 * Class:     org_opensearch_knn_jni_NmslibService
 * Method:    queryIndex
 * Signature: (J[FI)[Lorg/opensearch/knn/index/query/CustomKNNQueryResult;
 */
JNIEXPORT jobjectArray JNICALL Java_org_opensearch_knn_jni_NmslibService_queryIndex
  (JNIEnv *, jclass, jlong, jfloatArray, jint);

/*
 * Class:     org_opensearch_knn_jni_NmslibService
 * Method:    free
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_NmslibService_free
  (JNIEnv *, jclass, jlong);

/*
 * Class:     org_opensearch_knn_jni_NmslibService
 * Method:    initLibrary
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_NmslibService_initLibrary
  (JNIEnv *, jclass);

#ifdef __cplusplus
}
#endif
#endif
