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

#ifndef KNNPLUGIN_JNI_TESTS_NATIVE_STREAM_SUPPORT_UTIL_H_
#define KNNPLUGIN_JNI_TESTS_NATIVE_STREAM_SUPPORT_UTIL_H_

#include <stdexcept>
#include <fstream>

#include "test_util.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace test_util {

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

  int64_t remainingBytes() {
    return readTargetBytes.size() - nextReadIdx;
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



struct JavaFileIndexInputMock {
  JavaFileIndexInputMock(std::ifstream &_file_input, int32_t _buf_size)
      : file_input(_file_input),
        buffer(_buf_size) {
  }

  int64_t remainingBytes() {
    std::streampos currentPos = file_input.tellg();
    file_input.seekg(0, std::ios::end);
    std::streamsize fileSize = file_input.tellg();
    file_input.seekg(currentPos);
    return fileSize - currentPos;
  }

  int32_t copyBytes(int64_t read_size) {
    const auto copy_size = std::min((int64_t) buffer.size(), read_size);
    file_input.read(buffer.data(), copy_size);
    return (int32_t) copy_size;
  }

  std::ifstream &file_input;
  std::vector<char> buffer;
};  // struct JavaFileIndexInputMock



struct JavaFileIndexOutputMock {
  explicit JavaFileIndexOutputMock(const std::string &_file_path)
      : file_writer(_file_path, std::ios::out | std::ios::binary),
        buffer(64 * 1024) {
    file_writer.exceptions(std::ios::failbit | std::ios::badbit);
  }

  void writeBytes(int length) {
    file_writer.write(buffer.data(), length);
  }

  std::ofstream file_writer;
  std::vector<char> buffer;
};  // struct JavaFileIndexOutputMock

struct StreamIOError : public std::runtime_error {
  StreamIOError()
    : std::runtime_error(what()) {
  }

  const char* what() const noexcept final {
    return "Mocking IOError in Java side.";
  }
};  // struct StreamIOError

inline void setUpJavaFileOutputMocking(JavaFileIndexOutputMock &java_index_output,
                                       MockJNIUtil &mockJni,
                                       bool throwIOException) {
  EXPECT_CALL(mockJni, GetPrimitiveArrayCritical(::testing::_, ::testing::_, ::testing::_))
      .WillRepeatedly([&java_index_output](JNIEnv *env,
                                           jarray array,
                                           jboolean *isCopy) {
        return (jbyte *) java_index_output.buffer.data();
      });

  EXPECT_CALL(mockJni, CallNonvirtualVoidMethodA(::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_))
      .WillRepeatedly([&java_index_output](JNIEnv *env,
                                           jobject obj,
                                           jclass clazz,
                                           jmethodID methodID,
                                           jvalue *args) {
        const auto bytes_to_write = args[0].i;
        java_index_output.writeBytes(bytes_to_write);
      });

  EXPECT_CALL(mockJni, GetJavaBytesArrayLength(::testing::_, ::testing::_))
      .WillRepeatedly([&java_index_output](JNIEnv *env, jbyteArray arrayJ) {
        return java_index_output.buffer.size();
      });

  if (throwIOException) {
    EXPECT_CALL(mockJni, HasExceptionInStack(::testing::_, ::testing::_))
      .WillRepeatedly([](JNIEnv *env, const char* errorMsg){
        throw StreamIOError{};
      });
  } else {
    EXPECT_CALL(mockJni, HasExceptionInStack(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return());
  }
}

}  // namespace test_util

#endif //KNNPLUGIN_JNI_TESTS_NATIVE_STREAM_SUPPORT_UTIL_H_
