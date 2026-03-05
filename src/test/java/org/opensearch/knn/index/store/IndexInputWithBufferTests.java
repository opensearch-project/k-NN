/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.opensearch.knn.KNNTestCase;

import java.io.IOException;

public class IndexInputWithBufferTests extends KNNTestCase {

    public void testPeekFourcc() throws IOException {
        ByteBuffersDirectory dir = new ByteBuffersDirectory();
        try (IndexOutput out = dir.createOutput("test.bin", IOContext.DEFAULT)) {
            out.writeBytes(new byte[] { 'I', 'x', 'M', 'p' }, 4);
            out.writeBytes(new byte[] { 0, 0, 0, 0 }, 4); // extra bytes
        }
        try (IndexInput in = dir.openInput("test.bin", IOContext.DEFAULT)) {
            IndexInputWithBuffer buffer = new IndexInputWithBuffer(in);
            assertEquals("IxMp", buffer.peekFourcc());
            // Verify position is reset to 0
            assertEquals("IxMp", buffer.peekFourcc());
        }
    }

    public void testPeekFourccBinaryIndex() throws IOException {
        ByteBuffersDirectory dir = new ByteBuffersDirectory();
        try (IndexOutput out = dir.createOutput("test.bin", IOContext.DEFAULT)) {
            out.writeBytes(new byte[] { 'I', 'B', 'M', 'p' }, 4);
            out.writeBytes(new byte[] { 0, 0, 0, 0 }, 4);
        }
        try (IndexInput in = dir.openInput("test.bin", IOContext.DEFAULT)) {
            IndexInputWithBuffer buffer = new IndexInputWithBuffer(in);
            String fourcc = buffer.peekFourcc();
            assertEquals("IBMp", fourcc);
            assertTrue(fourcc.startsWith("IB"));
        }
    }
}
