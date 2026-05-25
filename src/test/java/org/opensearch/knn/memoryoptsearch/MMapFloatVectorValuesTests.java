/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import lombok.SneakyThrows;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.opensearch.knn.generate.SearchTestHelper;
import org.opensearch.knn.memoryoptsearch.faiss.MMapFloatVectorValues;
import org.opensearch.knn.memoryoptsearch.faiss.reconstruct.FaissQuantizerType;
import org.opensearch.knn.memoryoptsearch.faiss.vectorvalues.FaissFloatVectorValues;
import sun.misc.Unsafe;

import java.lang.reflect.Field;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public class MMapFloatVectorValuesTests extends LuceneTestCase {
    @SneakyThrows
    public void testValidBytesLoad() {
        // Creat temp dir
        final Path tempDirPath = createTempDir();
        final String fileName = "test.vec";
        final int numVectors = 100;
        final int dimension = 128;
        final int oneVectorByteSize = Float.BYTES * dimension;

        try (final Directory directory = new MMapDirectory(tempDirPath)) {
            // Write vectors
            final List<float[]> vectors = new ArrayList<>();
            try (final IndexOutput output = directory.createOutput(fileName, IOContext.DEFAULT)) {
                for (int i = 0; i < numVectors; i++) {
                    for (int j = 0; j < dimension; j++) {
                        final float[] vector = SearchTestHelper.generateOneSingleFloatVector(dimension, -2, 2, false);
                        vectors.add(vector);
                        for (final float val : vector) {
                            output.writeInt(Float.floatToIntBits(val));
                        }
                    }
                }
            }

            // Read validation
            try (final IndexInput input = directory.openInput(fileName, IOContext.DEFAULT)) {
                final long[] addressAndSize = MemorySegmentAddressExtractorUtil.tryExtractAddressAndSize(
                    input,
                    0,
                    numVectors * dimension * Float.BYTES
                );
                assertNotNull(addressAndSize);

                FaissFloatVectorValues faissFloatVectorValues = new FaissFloatVectorValues(input, oneVectorByteSize, dimension, numVectors);

                final MMapFloatVectorValues values = new MMapFloatVectorValues(faissFloatVectorValues, addressAndSize);

                // Ensure properties are correct.
                assertEquals(oneVectorByteSize, values.getVectorByteLength());
                assertEquals(addressAndSize, values.getAddressAndSize());

                // Ensure reads are correct
                // This will not be used in optimization path, but hnsw graph searcher will call this when traversing top level layers.
                for (int i = 0; i < numVectors; i++) {
                    final float[] rawFloatVector = values.vectorValue(i);
                    final float[] expectedVector = vectors.get(i);
                    compareFloats(rawFloatVector, expectedVector);
                }

                // Now, validating mmap read
                final Unsafe unsafe = getUnsafe();
                final byte[] buffer = new byte[oneVectorByteSize];
                final float[] acquiredVector = new float[dimension];
                long address = addressAndSize[0];

                for (int i = 0; i < numVectors; i++) {
                    // Per each vector, load and compare values
                    final float[] expectedVector = vectors.get(i);
                    for (int k = 0; k < oneVectorByteSize; ++k) {
                        buffer[k] = unsafe.getByte(address++);
                    }
                    FloatBuffer fb = ByteBuffer.wrap(buffer).order(ByteOrder.nativeOrder()).asFloatBuffer();
                    fb.get(acquiredVector);
                    compareFloats(acquiredVector, expectedVector);
                }
            }
        }
    }

    @SneakyThrows
    public void testConstructorWithQuantizerType() {
        final Path tempDirPath = createTempDir();
        final String fileName = "test_qt.vec";
        final int dimension = 4;
        final int oneVectorByteSize = Float.BYTES * dimension;

        try (final Directory directory = new MMapDirectory(tempDirPath)) {
            try (final IndexOutput output = directory.createOutput(fileName, IOContext.DEFAULT)) {
                for (int j = 0; j < dimension; j++) {
                    output.writeInt(Float.floatToIntBits(1.0f));
                }
            }

            try (final IndexInput input = directory.openInput(fileName, IOContext.DEFAULT)) {
                final long[] addressAndSize = MemorySegmentAddressExtractorUtil.tryExtractAddressAndSize(input, 0, oneVectorByteSize);
                assertNotNull(addressAndSize);

                FaissFloatVectorValues delegate = new FaissFloatVectorValues(input, oneVectorByteSize, dimension, 1);

                // Test 3-arg constructor with BF16 quantizer type
                final MMapFloatVectorValues values = new MMapFloatVectorValues(delegate, addressAndSize, FaissQuantizerType.QT_BF16);
                assertEquals(FaissQuantizerType.QT_BF16, values.getQuantizerType());
                assertEquals(dimension, values.dimension());
                assertEquals(1, values.size());

                // Test copy preserves quantizer type
                final MMapFloatVectorValues copied = (MMapFloatVectorValues) values.copy();
                assertEquals(FaissQuantizerType.QT_BF16, copied.getQuantizerType());

                // Test 2-arg constructor defaults to null
                final MMapFloatVectorValues valuesNoQt = new MMapFloatVectorValues(delegate, addressAndSize);
                assertNull(valuesNoQt.getQuantizerType());
            }
        }
    }

    // Unsafe is used for reading bytes directly from internal mapped pointer
    private static Unsafe getUnsafe() {
        try {
            Field f = Unsafe.class.getDeclaredField("theUnsafe");
            f.setAccessible(true);
            return (Unsafe) f.get(null);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private static void compareFloats(final float[] acquiredVectors, final float[] expected) {
        assertEquals(expected.length, acquiredVectors.length);

        for (int j = 0; j < expected.length; j++) {
            assertEquals(expected[j], acquiredVectors[j], 1e-6);
        }
    }
}
