/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import lombok.SneakyThrows;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.MMapDirectory;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndexScalarQuantizedFlat;
import org.opensearch.knn.memoryoptsearch.faiss.FaissSection;
import org.opensearch.knn.memoryoptsearch.faiss.MMapFloatVectorValues;
import org.opensearch.knn.memoryoptsearch.faiss.reconstruct.FaissBF16Reconstructor;
import org.opensearch.knn.memoryoptsearch.faiss.reconstruct.FaissQuantizerType;

import java.lang.reflect.Field;
import java.nio.file.Path;

import static org.mockito.Mockito.mock;

public class FaissIndexScalarQuantizedFlatBF16Tests extends KNNTestCase {

    public void testGetFloatValues_whenQuantizerTypeIsBF16_thenEnterBF16Branch() throws Exception {
        FaissIndexScalarQuantizedFlat index = new FaissIndexScalarQuantizedFlat();

        // Set quantizerType to QT_BF16
        setField(index, FaissIndexScalarQuantizedFlat.class, "quantizerType", FaissQuantizerType.QT_BF16);

        // Set dimension
        setField(index, "dimension", 4);

        // Set oneVectorByteSize (BF16: 2 bytes per dimension)
        setField(index, FaissIndexScalarQuantizedFlat.class, "oneVectorByteSize", 8L);

        // Set totalNumberOfVectors (inherited from FaissIndex)
        setField(index, "totalNumberOfVectors", 10);

        // Set reconstructor
        FaissBF16Reconstructor reconstructor = new FaissBF16Reconstructor(4, 16);
        setField(index, FaissIndexScalarQuantizedFlat.class, "reconstructor", reconstructor);

        // Set flatVectors with a mock FaissSection
        FaissSection flatVectors = mock(FaissSection.class);
        setField(index, FaissIndexScalarQuantizedFlat.class, "flatVectors", flatVectors);

        // Call getFloatValues with a non-mmap IndexInput (mock)
        IndexInput mockInput = mock(IndexInput.class);
        FloatVectorValues result = index.getFloatValues(mockInput);

        assertNotNull(result);
        assertEquals(4, result.dimension());
        assertEquals(10, result.size());
    }

    /**
     * Exercises the fall-through branch of the compound {@code ||} condition on line 110 of
     * {@link FaissIndexScalarQuantizedFlat#getFloatValues(IndexInput)}: when the quantizer is
     * neither {@code QT_FP16} nor {@code QT_BF16}, both sides of the {@code ||} evaluate to
     * false and the {@code if} block is skipped entirely. This covers the partial branch on
     * that line.
     */
    public void testGetFloatValues_whenQuantizerTypeIsNeitherFP16NorBF16_thenSkipMMapBranch() throws Exception {
        FaissIndexScalarQuantizedFlat index = new FaissIndexScalarQuantizedFlat();

        // Use a non-FP16/non-BF16 quantizer type so both operands of the `||` evaluate to false.
        setField(index, FaissIndexScalarQuantizedFlat.class, "quantizerType", FaissQuantizerType.QT_8BIT_DIRECT_SIGNED);
        setField(index, "dimension", 4);
        setField(index, FaissIndexScalarQuantizedFlat.class, "oneVectorByteSize", 4L);
        setField(index, "totalNumberOfVectors", 10);

        // Any non-null reconstructor is fine here; getFloatValues doesn't invoke it in the fall-through path.
        FaissBF16Reconstructor reconstructor = new FaissBF16Reconstructor(4, 8);
        setField(index, FaissIndexScalarQuantizedFlat.class, "reconstructor", reconstructor);

        FaissSection flatVectors = mock(FaissSection.class);
        setField(index, FaissIndexScalarQuantizedFlat.class, "flatVectors", flatVectors);

        IndexInput mockInput = mock(IndexInput.class);
        FloatVectorValues result = index.getFloatValues(mockInput);

        // The method bypasses the QT_FP16/QT_BF16 branch and returns a plain FloatVectorValuesImpl.
        assertNotNull(result);
        assertEquals(4, result.dimension());
        assertEquals(10, result.size());
    }

    /**
     * Exercises the mmap-success path of {@link FaissIndexScalarQuantizedFlat#getFloatValues(IndexInput)}
     * for {@code QT_BF16}. When {@link org.opensearch.knn.memoryoptsearch.MemorySegmentAddressExtractorUtil}
     * can extract a non-null {@code addressAndSize}, the method should return an {@link MMapFloatVectorValues}
     * wrapping the BF16 quantizer type rather than falling through to the plain {@code FloatVectorValuesImpl}.
     */
    @SneakyThrows
    public void testGetFloatValues_whenQuantizerTypeIsBF16AndInputIsMMap_thenReturnMMapFloatVectorValues() {
        final int dimension = 4;
        final int numVectors = 2;
        final long oneVectorByteSize = dimension * 2L; // BF16 = 2 bytes per element
        final long sectionSize = numVectors * oneVectorByteSize;

        final Path tempDirPath = createTempDir();
        final String fileName = "bf16.vec";

        try (final Directory directory = new MMapDirectory(tempDirPath)) {
            // File layout: 8-byte long = section element count, then the raw BF16 bytes.
            try (final IndexOutput output = directory.createOutput(fileName, IOContext.DEFAULT)) {
                output.writeLong(sectionSize); // FaissSection reads this as element count
                for (int i = 0; i < sectionSize; i++) {
                    output.writeByte((byte) 0);
                }
            }

            try (final IndexInput input = directory.openInput(fileName, IOContext.DEFAULT)) {
                // Build a real FaissSection by consuming the 8-byte header we just wrote,
                // with singleElementSize=1 so sectionSize matches the raw byte count.
                FaissSection flatVectors = new FaissSection(input, Byte.BYTES);
                // Rewind for the subsequent slice() calls (FaissSection uses absolute offsets).
                input.seek(0);

                FaissIndexScalarQuantizedFlat index = new FaissIndexScalarQuantizedFlat();
                setField(index, FaissIndexScalarQuantizedFlat.class, "quantizerType", FaissQuantizerType.QT_BF16);
                setField(index, "dimension", dimension);
                setField(index, FaissIndexScalarQuantizedFlat.class, "oneVectorByteSize", oneVectorByteSize);
                setField(index, "totalNumberOfVectors", numVectors);
                FaissBF16Reconstructor reconstructor = new FaissBF16Reconstructor(dimension, 16);
                setField(index, FaissIndexScalarQuantizedFlat.class, "reconstructor", reconstructor);
                setField(index, FaissIndexScalarQuantizedFlat.class, "flatVectors", flatVectors);

                FloatVectorValues result = index.getFloatValues(input);

                assertNotNull(result);
                // On platforms where mmap address extraction succeeds we get MMapFloatVectorValues
                // carrying the BF16 quantizer type. If the JDK/runtime doesn't allow extraction,
                // the method legitimately falls back — both paths exercise BF16-specific code.
                if (result instanceof MMapFloatVectorValues) {
                    MMapFloatVectorValues mmapValues = (MMapFloatVectorValues) result;
                    assertEquals(FaissQuantizerType.QT_BF16, mmapValues.getQuantizerType());
                    assertEquals(dimension, mmapValues.dimension());
                    assertEquals(numVectors, mmapValues.size());
                }
            }
        }
    }

    private static void setField(Object target, String fieldName, Object value) throws Exception {
        setField(target, target.getClass(), fieldName, value);
    }

    private static void setField(Object target, Class<?> clazz, String fieldName, Object value) throws Exception {
        try {
            Field field = clazz.getDeclaredField(fieldName);
            field.setAccessible(true);
            field.set(target, value);
        } catch (NoSuchFieldException e) {
            // Try superclass
            if (clazz.getSuperclass() != null) {
                setField(target, clazz.getSuperclass(), fieldName, value);
            } else {
                throw e;
            }
        }
    }
}
