// SPDX-License-Identifier: Apache-2.0
//
// The OpenSearch Contributors require contributions made to
// this file be licensed under the Apache-2.0 license or a
// compatible open source license.
//
// Modifications Copyright OpenSearch Contributors. See
// GitHub history for details.

#include "nmslib_wrapper.h"

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "jni_util.h"
#include "test_util.h"
#include "native_stream_support_util.h"

using ::test_util::JavaFileIndexInputMock;
using ::test_util::JavaFileIndexOutputMock;
using ::test_util::MockJNIUtil;
using ::test_util::StreamIOError;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::_;

void setUpJavaFileInputMocking(JavaFileIndexInputMock &java_index_input, MockJNIUtil &mockJni) {
  // Set up mocking values + mocking behavior in a method.
  EXPECT_CALL(mockJni, CallNonvirtualIntMethodA(_, _, _, _, _))
      .WillRepeatedly([&java_index_input](JNIEnv *env,
                                                 jobject obj,
                                                 jclass clazz,
                                                 jmethodID methodID,
                                                 jvalue *args) {
        return java_index_input.copyBytes(args[0].j);
      });
  EXPECT_CALL(mockJni, CallNonvirtualLongMethodA(_, _, _, _, _))
      .WillRepeatedly([&java_index_input](JNIEnv *env,
                                                 jobject obj,
                                                 jclass clazz,
                                                 jmethodID methodID,
                                                 jvalue *args) {
        return java_index_input.remainingBytes();
      });
  EXPECT_CALL(mockJni, GetPrimitiveArrayCritical(_, _, _))
      .WillRepeatedly([&java_index_input](JNIEnv *env,
                                                 jarray array,
                                                 jboolean *isCopy) {
        return (jbyte *) java_index_input.buffer.data();
      });
  EXPECT_CALL(mockJni, ReleasePrimitiveArrayCritical(_, _, _, _)).WillRepeatedly(Return());
}

TEST(NmslibStreamLoadingTest, BasicAssertions) {
  for (auto throwIOException : std::array<bool, 2> {false, true}) {
      // Initialize nmslib
      similarity::initLibrary();

      // Define index data
      int numIds = 100;
      std::vector<int> ids;
      auto vectors = new std::vector<float>();
      int dim = 2;
      vectors->reserve(dim * numIds);
      for (int i = 0; i < numIds; ++i) {
        ids.push_back(i);
        for (int j = 0; j < dim; ++j) {
          vectors->push_back(test_util::RandomFloat(-500.0, 500.0));
        }
      }

      std::string spaceType = knn_jni::L2;
      std::string indexPath = test_util::RandomString(
          10, "/tmp/", ".nmslib");

      std::unordered_map<std::string, jobject> parametersMap;
      int efConstruction = 512;
      int m = 96;

      parametersMap[knn_jni::SPACE_TYPE] = (jobject) &spaceType;
      parametersMap[knn_jni::EF_CONSTRUCTION] = (jobject) &efConstruction;
      parametersMap[knn_jni::M] = (jobject) &m;

      // Set up jni
      NiceMock<JNIEnv> jniEnv;

      NiceMock<MockJNIUtil> mockJNIUtil;
      JavaFileIndexOutputMock javaFileIndexOutputMock {indexPath};
      setUpJavaFileOutputMocking(javaFileIndexOutputMock, mockJNIUtil, throwIOException);
      knn_jni::stream::NativeEngineIndexOutputMediator mediator {&mockJNIUtil, &jniEnv, (jobject) (&javaFileIndexOutputMock)};
      knn_jni::stream::NmslibOpenSearchIOWriter writer {&mediator};

      EXPECT_CALL(mockJNIUtil,
                  GetJavaObjectArrayLength(
                      &jniEnv, reinterpret_cast<jobjectArray>(vectors)))
          .WillRepeatedly(Return(vectors->size()));

      EXPECT_CALL(mockJNIUtil,
                  GetJavaIntArrayLength(&jniEnv, reinterpret_cast<jintArray>(&ids)))
          .WillRepeatedly(Return(ids.size()));

      EXPECT_CALL(mockJNIUtil,
                  ConvertJavaMapToCppMap(&jniEnv, reinterpret_cast<jobject>(&parametersMap)))
          .WillRepeatedly(Return(parametersMap));

      // Create the index
      try {
          knn_jni::nmslib_wrapper::CreateIndex(
              &mockJNIUtil, &jniEnv, reinterpret_cast<jintArray>(&ids),
              (jlong) vectors, dim, (jobject) (&javaFileIndexOutputMock),
              (jobject) &parametersMap);
          javaFileIndexOutputMock.file_writer.close();
      } catch (const StreamIOError& e) {
          continue;
      }

      // Create Java index input mock.
      std::ifstream file_input{indexPath, std::ios::binary};
      const int32_t buffer_size = 128;
      JavaFileIndexInputMock java_file_index_input_mock{file_input, buffer_size};
      setUpJavaFileInputMocking(java_file_index_input_mock, mockJNIUtil);

      // Make sure index can be loaded
      jlong index = knn_jni::nmslib_wrapper::LoadIndexWithStream(
          &mockJNIUtil, &jniEnv,
          (jobject) (&java_file_index_input_mock),
          (jobject) (&parametersMap));

      knn_jni::nmslib_wrapper::Free(index);

      // Clean up
      file_input.close();
      std::remove(indexPath.c_str());
  }  // End for
}
