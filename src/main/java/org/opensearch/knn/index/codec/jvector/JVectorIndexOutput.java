/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import io.github.jbellis.jvector.disk.RandomAccessWriter;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.store.IndexOutput;

import java.io.IOException;

@Log4j2
public class JVectorIndexOutput extends IndexOutput {
    private final RandomAccessWriter randomAccessWriter;

    public JVectorIndexOutput(RandomAccessWriter randomAccessWriter) {
        super("JVectorIndexOutput", "JVectorIndexOutput");
        this.randomAccessWriter = randomAccessWriter;
    }

    @Override
    public void close() throws IOException {
        randomAccessWriter.close();
    }

    @Override
    public long getFilePointer() {
        try {
            return randomAccessWriter.position();
        } catch (IOException e) {
            log.error("Error getting file pointer", e);
            throw new RuntimeException(e);
        }
    }

    @Override
    public long getChecksum() throws IOException {
        return randomAccessWriter.checksum(0, randomAccessWriter.position());
    }

    @Override
    public void writeByte(byte b) throws IOException {
        randomAccessWriter.write(b);
    }

    @Override
    public void writeBytes(byte[] b, int offset, int length) throws IOException {
        randomAccessWriter.write(b, offset, length);
    }
}
