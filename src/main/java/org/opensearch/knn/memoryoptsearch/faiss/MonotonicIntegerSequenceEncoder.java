/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import lombok.experimental.UtilityClass;
import org.apache.lucene.store.ByteBuffersDataOutput;
import org.apache.lucene.store.ByteBuffersIndexOutput;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.packed.DirectMonotonicReader;
import org.apache.lucene.util.packed.DirectMonotonicWriter;
import org.opensearch.common.lucene.store.ByteArrayIndexInput;

import java.io.IOException;

@UtilityClass
public class MonotonicIntegerSequenceEncoder {
    // Use 64KB (=2^16) block size for monotonic encoding.
    private static final int DIRECT_MONOTONIC_BLOCK_SHIFT = 16;

    /**
     * Encodes a monotonically increasing sequence of integers and returns a decoder. During encoding, it reads long values from the
     * provided {@link IndexInput} and converts them to integers. If a value exceeds {@link Integer#MAX_VALUE},
     * an {@link ArithmeticException} is thrown. If the sequence is not strictly monotonically increasing,
     * an {@link IllegalArgumentException} is thrown.
     *
     * @param numElements Number of elements in sequence for encoding.
     * @param input Input stream for integer sequence.
     * @return A decoder to return a mapped integer with given index i.
     * @throws IOException
     */
    public static DirectMonotonicReader encode(final int numElements, final IndexInput input) throws IOException {
        // Prepare a buffer for meta
        ByteBuffersDataOutput dataOutput = new ByteBuffersDataOutput();
        ByteBuffersIndexOutput dataIndexOutput = new ByteBuffersIndexOutput(
            dataOutput,
            "MonotonicSequenceEncoder",
            "MonotonicSequenceEncoderData"
        );

        // Prepare a buffer for data
        ByteBuffersDataOutput metaOutput = new ByteBuffersDataOutput();
        ByteBuffersIndexOutput metaIndexOutput = new ByteBuffersIndexOutput(
            metaOutput,
            "MonotonicSequenceEncoder",
            "MonotonicSequenceEncoderMeta"
        );

        // Prepare an encoder with a 64KB(=2^16) chunk.
        DirectMonotonicWriter encoder = DirectMonotonicWriter.getInstance(
            metaIndexOutput,
            dataIndexOutput,
            numElements,
            DIRECT_MONOTONIC_BLOCK_SHIFT
        );

        // Encode integer sequence.
        boolean isIdenticalMapping = true;
        for (long i = 0; i < numElements; i++) {
            final long value = Math.toIntExact(input.readLong());
            if (value != i) {
                isIdenticalMapping = false;
            }
            encoder.add(value);
        }

        encoder.finish();

        // Close outputs
        IOUtils.close(dataIndexOutput, metaIndexOutput);

        if (isIdenticalMapping) {
            // It's an identical mapping (e.g. i -> i), no need to continue encoding.
            return null;
        }

        // Create input streams for both meta, data
        final byte[] metaBytes = metaOutput.toArrayCopy();
        final IndexInput metaInput = new ByteArrayIndexInput("MonotonicSequenceEncoder", metaBytes);

        final byte[] dataBytes = dataOutput.toArrayCopy();
        final ByteArrayIndexInput dataInput = new ByteArrayIndexInput("MonotonicSequenceEncoder", dataBytes);

        // Create decoder
        final DirectMonotonicReader.Meta encodingMeta = DirectMonotonicReader.loadMeta(
            metaInput,
            numElements,
            DIRECT_MONOTONIC_BLOCK_SHIFT
        );
        return DirectMonotonicReader.getInstance(encodingMeta, dataInput);
    }
}
