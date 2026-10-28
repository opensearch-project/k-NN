/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import lombok.SneakyThrows;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.index.MergeAbortChecker;
import org.mockito.MockedStatic;
import org.opensearch.common.lucene.store.ByteArrayIndexInput;
import org.opensearch.knn.common.exception.TerminalIOException;
import org.opensearch.knn.index.codec.nativeindex.IndexBuildAbortedException;
import org.opensearch.knn.KNNTestCase;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.Arrays;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doCallRealMethod;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.verify;

public class IndexOutputWithBufferTests extends KNNTestCase {

    public void testCopyFromPreservesBytesAndTracksRemainingLength() throws IOException {
        byte[] bytes = new byte[128 * 1024 + 17];
        random().nextBytes(bytes);
        try (Directory directory = newDirectory(); IndexInput input = new ByteArrayIndexInput("copy", bytes)) {
            try (IndexOutputWithBuffer output = new IndexOutputWithBuffer(directory.createOutput("copy.faiss", IOContext.DEFAULT))) {
                input.seek(7);
                output.copyFrom(input);
                assertEquals(bytes.length - 7, output.getBytesWritten());
            }
            try (IndexInput result = directory.openInput("copy.faiss", IOContext.DEFAULT)) {
                byte[] actual = new byte[Math.toIntExact(result.length())];
                result.readBytes(actual, 0, actual.length);
                assertArrayEquals(Arrays.copyOfRange(bytes, 7, bytes.length), actual);
            }
        }
    }

    public void testCopyReadFailurePreservesSuccessfulWriteCount() throws IOException {
        IOException failure = new IOException("Read failed after one chunk");
        IndexOutput delegate = mock(IndexOutput.class);
        try (
            IndexInput input = spy(new ByteArrayIndexInput("copy", new byte[128 * 1024]));
            IndexOutputWithBuffer output = new IndexOutputWithBuffer(delegate)
        ) {
            doCallRealMethod().doThrow(failure).when(input).readBytes(any(byte[].class), eq(0), eq(64 * 1024));
            assertSame(failure, expectThrows(IOException.class, () -> output.copyFrom(input)));
            assertEquals(64 * 1024, output.getBytesWritten());
            verify(delegate).writeBytes(any(byte[].class), eq(0), eq(64 * 1024));
        }
    }

    public void testCopyWriteFailureIsTerminal() throws IOException {
        IOException failure = new IOException("Output failed");
        IndexOutput delegate = mock(IndexOutput.class);
        doThrow(failure).when(delegate).writeBytes(any(byte[].class), eq(0), eq(1));
        try (
            IndexInput input = new ByteArrayIndexInput("copy", new byte[1]);
            IndexOutputWithBuffer output = new IndexOutputWithBuffer(delegate)
        ) {
            TerminalIOException exception = expectThrows(TerminalIOException.class, () -> output.copyFrom(input));
            assertSame(failure, exception.getCause());
            assertEquals(0, output.getBytesWritten());
        }
    }

    public void testCopyHonorsCancellationBetweenChunks() throws IOException {
        IndexOutput delegate = mock(IndexOutput.class);
        try (
            MockedStatic<MergeAbortChecker> mergeAbort = mockStatic(MergeAbortChecker.class);
            IndexInput input = new ByteArrayIndexInput("copy", new byte[128 * 1024]);
            IndexOutputWithBuffer output = new IndexOutputWithBuffer(delegate)
        ) {
            mergeAbort.when(MergeAbortChecker::isMergeAborted).thenReturn(false, true);
            expectThrows(IndexBuildAbortedException.class, () -> output.copyFrom(input));
            assertEquals(64 * 1024, output.getBytesWritten());
            verify(delegate).writeBytes(any(byte[].class), eq(0), eq(64 * 1024));
        }
    }

    @SneakyThrows
    public void testWriteFromStream_thenTracksBytesWritten() {
        try (Directory directory = newDirectory()) {
            IndexOutput output = directory.createOutput("test.faiss", IOContext.DEFAULT);
            IndexOutputWithBuffer buffer = new IndexOutputWithBuffer(output);

            assertEquals(0, buffer.getBytesWritten());

            byte[] data = new byte[] { 1, 2, 3, 4, 5, 6, 7, 8 };
            buffer.writeFromStreamWithBuffer(new ByteArrayInputStream(data), data.length);

            assertEquals(data.length, buffer.getBytesWritten());

            buffer.close();
        }
    }

    @SneakyThrows
    public void testWriteBytes_thenTracksBytesWritten() {
        try (Directory directory = newDirectory()) {
            IndexOutput output = directory.createOutput("test.faiss", IOContext.DEFAULT);
            IndexOutputWithBuffer buffer = new IndexOutputWithBuffer(output);

            assertEquals(0, buffer.getBytesWritten());

            // writeBytes is called from JNI — it reads from the internal buffer
            // We can't easily fill the internal buffer without JNI, but we can verify 0-length write
            buffer.writeBytes(0);
            assertEquals(0, buffer.getBytesWritten());

            buffer.close();
        }
    }

    @SneakyThrows
    public void testReset_whenBytesWritten_thenClearsFileAndResetsBytesWritten() {
        try (Directory directory = newDirectory()) {
            final String fileName = "test_reset.faiss";
            IndexOutput output = directory.createOutput(fileName, IOContext.DEFAULT);
            IndexOutputWithBuffer buffer = new IndexOutputWithBuffer(output);

            // Write some data
            byte[] data = new byte[] { 0x49, 0x78, 0x4d, 0x70, 0x00, 0x04, 0x00, 0x00 };
            buffer.writeFromStreamWithBuffer(new ByteArrayInputStream(data), data.length);
            assertEquals(data.length, buffer.getBytesWritten());

            // Reset should close old file, delete it, and create a fresh one
            buffer.reset(directory, IOContext.DEFAULT);

            assertEquals(0, buffer.getBytesWritten());
            assertEquals(fileName, buffer.getName());

            // Write to the new output — should work without error
            byte[] newData = new byte[] { 1, 2, 3 };
            buffer.writeFromStreamWithBuffer(new ByteArrayInputStream(newData), newData.length);
            assertEquals(newData.length, buffer.getBytesWritten());

            buffer.close();
            assertEquals(newData.length, directory.fileLength(fileName));
        }
    }

    @SneakyThrows
    public void testReset_whenNothingWritten_thenStillWorks() {
        try (Directory directory = newDirectory()) {
            final String fileName = "test_reset_empty.faiss";
            IndexOutput output = directory.createOutput(fileName, IOContext.DEFAULT);
            IndexOutputWithBuffer buffer = new IndexOutputWithBuffer(output);

            assertEquals(0, buffer.getBytesWritten());

            // Reset on empty file — should not throw
            buffer.reset(directory, IOContext.DEFAULT);

            assertEquals(0, buffer.getBytesWritten());
            assertEquals(fileName, buffer.getName());

            buffer.close();
        }
    }

    @SneakyThrows
    public void testGetName_thenReturnsFileName() {
        try (Directory directory = newDirectory()) {
            final String fileName = "my_field.faiss";
            IndexOutput output = directory.createOutput(fileName, IOContext.DEFAULT);
            IndexOutputWithBuffer buffer = new IndexOutputWithBuffer(output);

            assertEquals(fileName, buffer.getName());

            buffer.close();
        }
    }

    @SneakyThrows
    public void testWriteFooter_thenWritesCodecFooter() {
        try (Directory directory = newDirectory()) {
            final String fileName = "footer_test.faiss";
            IndexOutput output = directory.createOutput(fileName, IOContext.DEFAULT);
            IndexOutputWithBuffer buffer = new IndexOutputWithBuffer(output);

            // Write some data first (footer needs non-zero content for CRC)
            byte[] data = new byte[] { 1, 2, 3, 4 };
            buffer.writeFromStreamWithBuffer(new ByteArrayInputStream(data), data.length);

            buffer.writeFooter();
            buffer.close();

            // Lucene codec footer is 16 bytes (magic + algorithmID + CRC)
            assertEquals(data.length + 16, directory.fileLength(fileName));
        }
    }

    @SneakyThrows
    public void testClose_thenClosesUnderlyingOutput() {
        try (Directory directory = newDirectory()) {
            final String fileName = "close_test.faiss";
            IndexOutput output = directory.createOutput(fileName, IOContext.DEFAULT);
            IndexOutputWithBuffer buffer = new IndexOutputWithBuffer(output);

            buffer.close();

            // Writing after close should throw
            expectThrows(Exception.class, () -> buffer.writeFromStreamWithBuffer(new ByteArrayInputStream(new byte[] { 1 }), 1));
        }
    }

    @SneakyThrows
    public void testAutoCloseable_withTryWithResources() {
        try (Directory directory = newDirectory()) {
            final String fileName = "autocloseable_test.faiss";
            try (IndexOutputWithBuffer buffer = new IndexOutputWithBuffer(directory.createOutput(fileName, IOContext.DEFAULT))) {
                byte[] data = new byte[] { 10, 20, 30 };
                buffer.writeFromStreamWithBuffer(new ByteArrayInputStream(data), data.length);
                buffer.writeFooter();
            }
            // File should exist and have content after try-with-resources closes it
            assertEquals(3 + 16, directory.fileLength(fileName));
        }
    }
}
