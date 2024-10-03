// SPDX-License-Identifier: Apache-2.0
//
// The OpenSearch Contributors require contributions made to
// this file be licensed under the Apache-2.0 license or a
// compatible open source license.
//
// Modifications Copyright OpenSearch Contributors. See
// GitHub history for details.

#include "native_engines_stream_support.h"
#include <gmock/gmock.h>
#include "test_util.h"
#include <jni.h>
#include <random>
#include <algorithm>
#include <vector>

using ::testing::Return;
using knn_jni::stream::FaissOpenSearchIOReader;
using knn_jni::stream::NativeEngineIndexInputMediator;
using test_util::MockJNIUtil;

// Mocking IndexInputWithBuffer.
struct JavaIndexInputMock {
  JavaIndexInputMock(std::string _readTargetBytes, int32_t _bufSize)
      : readTargetBytes(std::move(_readTargetBytes)),
        nextReadIdx(),
        buffer(_bufSize) {
  }

  // This method is simulating `copyBytes` in IndexInputWithBuffer.
  int32_t simulateCopyReads(int64_t readBytes) {
    readBytes = std::min(readBytes, (int64_t) buffer.size());
    readBytes = std::min(readBytes, (int64_t) (readTargetBytes.size() - nextReadIdx));
    std::memcpy(buffer.data(), readTargetBytes.data() + nextReadIdx, readBytes);
    nextReadIdx += readBytes;
    return (int32_t) readBytes;
  }

  static std::string makeRandomBytes(int32_t bytesSize) {
    // Define the list of possible characters
    static const string CHARACTERS
        = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuv"
          "wxyz0123456789";

    // Create a random number generator
    std::random_device rd;
    std::mt19937 generator(rd());

    // Create a distribution to uniformly select from all characters
    std::uniform_int_distribution<> distribution(
        0, CHARACTERS.size() - 1);

    // Pre-allocate the string with the desired length
    std::string randomString(bytesSize, '\0');

    // Use generate_n with a back_inserter iterator
    std::generate_n(randomString.begin(), bytesSize, [&]() {
      return CHARACTERS[distribution(generator)];
    });

    return randomString;
  }

  std::string readTargetBytes;
  int64_t nextReadIdx;
  std::vector<char> buffer;
};  // struct JavaIndexInputMock

void setUpMockJNIUtil(JavaIndexInputMock &javaIndexInputMock, MockJNIUtil &mockJni) {
  // Set up mocking values + mocking behavior in a method.
  ON_CALL(mockJni, FindClassFromJNIEnv).WillByDefault(Return((jclass) 1));
  ON_CALL(mockJni, GetMethodID).WillByDefault(Return((jmethodID) 1));
  ON_CALL(mockJni, GetFieldID).WillByDefault(Return((jfieldID) 1));
  ON_CALL(mockJni, GetObjectField).WillByDefault(Return((jobject) 1));
  ON_CALL(mockJni, CallIntMethodLong).WillByDefault([&javaIndexInputMock](JNIEnv *env,
                                                                          jobject obj,
                                                                          jmethodID methodID,
                                                                          int64_t longArg) {
    return javaIndexInputMock.simulateCopyReads(longArg);
  });
  ON_CALL(mockJni, GetPrimitiveArrayCritical).WillByDefault([&javaIndexInputMock](JNIEnv *env,
                                                                                  jarray array,
                                                                                  jboolean *isCopy) {
    return (jbyte *) javaIndexInputMock.buffer.data();
  });
  ON_CALL(mockJni, ReleasePrimitiveArrayCritical).WillByDefault(Return());
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
    MockJNIUtil mockJni;
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
