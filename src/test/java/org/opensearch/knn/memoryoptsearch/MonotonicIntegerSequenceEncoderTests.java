/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import lombok.SneakyThrows;
import org.apache.lucene.store.ByteBuffersDataOutput;
import org.apache.lucene.store.ByteBuffersIndexOutput;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.packed.DirectMonotonicReader;
import org.opensearch.common.lucene.store.ByteArrayIndexInput;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.memoryoptsearch.faiss.MonotonicIntegerSequenceEncoder;

import java.io.IOException;
import java.util.function.Function;

public class MonotonicIntegerSequenceEncoderTests extends KNNTestCase {
    @SneakyThrows
    public void testEncodeIdentityCase() {
        // Create a sequence
        final int size = 100000;
        final IndexInput input = createSequence((i) -> i, size);

        // Call encode
        DirectMonotonicReader reader = MonotonicIntegerSequenceEncoder.encode(size, input);

        // It should be null, as it's identical mapping
        assertNull(reader);
    }

    @SneakyThrows
    public void testEncodeIdentityCaseSmallVolume() {
        // Create a sequence
        final int size = 10;
        final IndexInput input = createSequence((i) -> i, size);

        // Call encode
        DirectMonotonicReader reader = MonotonicIntegerSequenceEncoder.encode(size, input);

        // It should be null, as it's identical mapping
        assertNull(reader);
    }

    @SneakyThrows
    public void testEmptyCase() {
        DirectMonotonicReader reader = MonotonicIntegerSequenceEncoder.encode(0, null);
        assertNull(reader);
    }

    @SneakyThrows
    public void testTooBigNumber() {
        final int size = 100;

        final IndexInput input = createSequence((i) -> Integer.MAX_VALUE + i, size);

        try {
            // Call encode
            MonotonicIntegerSequenceEncoder.encode(size, input);
            fail();
        } catch (ArithmeticException e) {}
    }

    @SneakyThrows
    public void testWhenNegativeNumbers() {
        final int size = 100;

        final IndexInput input = createSequence((i) -> -i, size);

        try {
            // Call encode
            MonotonicIntegerSequenceEncoder.encode(size, input);
            fail();
        } catch (IllegalArgumentException e) {}
    }

    @SneakyThrows
    public void testIncreasingSequence() {
        // Create a sequence
        final int size = 100000;
        final IndexInput input = createSequence((i) -> i / 5, size);

        // Call encode
        DirectMonotonicReader reader = MonotonicIntegerSequenceEncoder.encode(size, input);

        // It should not be null
        assertNotNull(reader);

        // Validate values
        for (int i = 0; i < size; ++i) {
            assertEquals(i / 5, reader.get(i));
        }
    }

    @SneakyThrows
    public void testIncreasingSequenceWithSmallVolume() {
        // Create a sequence
        final int size = 100;
        final IndexInput input = createSequence((i) -> i / 5, size);

        // Call encode
        DirectMonotonicReader reader = MonotonicIntegerSequenceEncoder.encode(size, input);

        // It should not be null
        assertNotNull(reader);

        // Validate values
        for (int i = 0; i < size; ++i) {
            assertEquals(i / 5, reader.get(i));
        }
    }

    private static IndexInput createSequence(Function<Long, Long> mapping, int length) throws IOException {
        ByteBuffersDataOutput dataOutput = new ByteBuffersDataOutput();
        ByteBuffersIndexOutput dataIndexOutput = new ByteBuffersIndexOutput(
            dataOutput,
            "MonotonicIntegerSequenceEncoderTests",
            "MonotonicIntegerSequenceEncoderTests"
        );

        for (int i = 0; i < length; i++) {
            dataIndexOutput.writeLong(mapping.apply((long) i));
        }

        dataIndexOutput.close();
        byte[] bytes = dataOutput.toArrayCopy();
        return new ByteArrayIndexInput("MonotonicIntegerSequenceEncoderTests", bytes);
    }
}
