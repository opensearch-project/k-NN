// SPDX-License-Identifier: Apache-2.0
//
// The OpenSearch Contributors require contributions made to
// this file be licensed under the Apache-2.0 license or a
// compatible open source license.
//
// Modifications Copyright OpenSearch Contributors. See
// GitHub history for details.

#include "faiss_stream_support.h"
#include "native_stream_support_util.h"
#include "test_util.h"

#include <gmock/gmock.h>
#include <jni.h>
#include <algorithm>
#include <vector>

using ::testing::_;
using ::testing::Return;
using knn_jni::stream::FaissOpenSearchIOReader;
using knn_jni::stream::NativeEngineIndexInputMediator;
using test_util::MockJNIUtil;
using test_util::JavaIndexInputMock;
using ::testing::NiceMock;
using ::testing::Return;

void setUpMockJNIUtil(JavaIndexInputMock &javaIndexInputMock, MockJNIUtil &mockJni) {
  // Set up mocking values + mocking behavior in a method.
  EXPECT_CALL(mockJni, CallNonvirtualIntMethodA(_, _, _, _, _))
      .WillRepeatedly([&javaIndexInputMock](JNIEnv *env,
                                            jobject obj,
                                            jclass clazz,
                                            jmethodID methodID,
                                            jvalue* args) {
        return javaIndexInputMock.simulateCopyReads(args[0].j);
      });
  EXPECT_CALL(mockJni, CallNonvirtualLongMethodA(_, _, _, _, _))
      .WillRepeatedly([&javaIndexInputMock](JNIEnv *env,
                                            jobject obj,
                                            jclass clazz,
                                            jmethodID methodID,
                                            jvalue* args) {
        return javaIndexInputMock.remainingBytes();
      });
  EXPECT_CALL(mockJni, GetPrimitiveArrayCritical(_, _, _))
      .WillRepeatedly([&javaIndexInputMock](JNIEnv *env,
                                            jarray array,
                                            jboolean *isCopy) {
        return (jbyte *) javaIndexInputMock.buffer.data();
      });
  EXPECT_CALL(mockJni, ReleasePrimitiveArrayCritical(_, _, _, _))
      .WillRepeatedly(Return());
}

TEST(FaissStreamSupportTest, NativeEngineIndexInputMediatorCopyWhenEmpty) {
  for (auto contentSize : std::vector<int32_t>{0, 2222, 7777, 1024, 77, 1}) {
    // Set up mockings
    MockJNIUtil mockJni;
    JavaIndexInputMock javaIndexInputMock{
        JavaIndexInputMock::makeRandomBytes(contentSize), 1024};
    setUpMockJNIUtil(javaIndexInputMock, mockJni);

    // Prepare copying
    NativeEngineIndexInputMediator mediator{&mockJni, nullptr, nullptr};
    std::string readBuffer(javaIndexInputMock.readTargetBytes.size(), '\0');

    // Call copyBytes
    mediator.copyBytes((int32_t) javaIndexInputMock.readTargetBytes.size(), (uint8_t *) readBuffer.data());

    // Expected that we acquired the same contents as readTargetBytes
    ASSERT_EQ(javaIndexInputMock.readTargetBytes, readBuffer);
  }  // End for
}

TEST(FaissStreamSupportTest, FaissOpenSearchIOReaderCopy) {
  for (auto contentSize : std::vector<int32_t>{0, 2222, 7777, 1024, 77, 1}) {
    // Set up mockings
    NiceMock<MockJNIUtil> mockJni;
    JavaIndexInputMock javaIndexInputMock{
        JavaIndexInputMock::makeRandomBytes(contentSize), 1024};
    setUpMockJNIUtil(javaIndexInputMock, mockJni);

    // Prepare copying
    NativeEngineIndexInputMediator mediator{&mockJni, nullptr, nullptr};
    std::string readBuffer;
    readBuffer.resize(javaIndexInputMock.readTargetBytes.size());
    FaissOpenSearchIOReader ioReader{&mediator};

    // Read bytes
    const auto readBytes =
        ioReader((void *) readBuffer.data(), 1, javaIndexInputMock.readTargetBytes.size());

    // Expected that we acquired the same contents as readTargetBytes
    ASSERT_EQ(javaIndexInputMock.readTargetBytes.size(), readBytes);
    ASSERT_EQ(javaIndexInputMock.readTargetBytes, readBuffer);
  }  // End for
}
