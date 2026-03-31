/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import lombok.SneakyThrows;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.store.ByteBuffersDataOutput;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.packed.DirectMonotonicReader;
import org.mockito.MockedStatic;
import org.mockito.stubbing.Answer;
import org.opensearch.common.lucene.store.ByteArrayIndexInput;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.memoryoptsearch.faiss.FaissHNSWIndex;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIdMapIndex;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndex;
import org.opensearch.knn.memoryoptsearch.faiss.binary.FaissBinaryIndex;
import org.opensearch.knn.memoryoptsearch.faiss.vectorvalues.FaissByteVectorValues;
import org.opensearch.knn.memoryoptsearch.faiss.vectorvalues.FaissFloatVectorValues;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicInteger;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.when;

public class FaissIdMapIndexTests extends KNNTestCase {
    public static final int CODE_SIZE = 64;

    public void testLoadDenseCase() {
        doTestLoadDenseCase(FaissIdMapIndex.IXMP);
        doTestLoadDenseCase(FaissIdMapIndex.IBMP);
    }

    private void doTestLoadDenseCase(final String indexType) {
        // Dimension : 128
        // #Vectors : 100
        // Metric : L2
        final int totalNumberOfVectors = 100;
        final int dimension = 128;
        final boolean l2Metric = true;

        // Prepare id mapping
        final long[] mappingTable = new long[totalNumberOfVectors];
        for (int i = 0; i < totalNumberOfVectors; ++i) {
            mappingTable[i] = i;
        }

        // Load index
        final FaissIdMapIndex index = triggerLoadAndGetIndex(dimension, totalNumberOfVectors, l2Metric, mappingTable, indexType);

        // We expect it to be null when identity case.
        final long[] loadedMappingTable = getVectorIdToDocIdMapping(index, 0);
        assertNull(loadedMappingTable);

        // Validate common header
        validateHeader(indexType, index, dimension, totalNumberOfVectors);
    }

    public void testLoadSparseCase() {
        doTestLoadSparseCase(FaissIdMapIndex.IXMP);
        doTestLoadSparseCase(FaissIdMapIndex.IBMP);
    }

    @SneakyThrows
    private void doTestLoadSparseCase(final String indexType) {
        // Dimension : 128
        // #Vectors : 100
        // Metric : L2
        final int totalNumberOfVectors = 100;
        final int dimension = 128;
        final boolean l2Metric = true;

        // Prepare id mapping
        // Assuming only 0th, 3rd, 6th, ..., 3k_th docs have KNN field.
        final long[] mappingTable = new long[totalNumberOfVectors];
        for (int i = 0; i < totalNumberOfVectors; ++i) {
            mappingTable[i] = 3 * i;
        }

        // Load index
        final FaissIdMapIndex index = triggerLoadAndGetIndex(dimension, totalNumberOfVectors, l2Metric, mappingTable, indexType);

        // We expect it to be null when identity case.
        final long[] loadedMappingTable = getVectorIdToDocIdMapping(index, totalNumberOfVectors);
        assertNotNull(loadedMappingTable);
        assertEquals(totalNumberOfVectors, loadedMappingTable.length);
        assertArrayEquals(mappingTable, loadedMappingTable);

        // Sparse byte vectors
        final ByteVectorValues byteVectorValues = index.getByteValues(null);
        Bits bitsFromByteVectors = byteVectorValues.getAcceptOrds(null);
        assertNull(bitsFromByteVectors);

        // Verify that getAcceptOrds correctly maps vector ordinal -> doc ID via acceptDocs
        final Bits mockAcceptDocsForBytes = mock(Bits.class);
        final AtomicInteger byteIdx = new AtomicInteger(0);
        doAnswer((Answer<Boolean>) invocation -> {
            final int docId = invocation.getArgument(0);
            assertEquals(mappingTable[byteIdx.getAndIncrement()], docId);
            return true;
        }).when(mockAcceptDocsForBytes).get(anyInt());

        bitsFromByteVectors = byteVectorValues.getAcceptOrds(mockAcceptDocsForBytes);
        assertEquals(totalNumberOfVectors, bitsFromByteVectors.length());
        for (int i = 0; i < totalNumberOfVectors; ++i) {
            bitsFromByteVectors.get(i);
            assertEquals(mappingTable[i], byteVectorValues.ordToDoc(i));
        }
        assertEquals(totalNumberOfVectors, byteIdx.get());

        // Sparse float vectors
        final FloatVectorValues floatVectorValues = index.getFloatValues(null);
        Bits bitsFromFloatVectors = floatVectorValues.getAcceptOrds(null);
        assertNull(bitsFromFloatVectors);

        final Bits mockAcceptDocsForFloats = mock(Bits.class);
        final AtomicInteger floatIdx = new AtomicInteger(0);
        doAnswer((Answer<Boolean>) invocation -> {
            final int docId = invocation.getArgument(0);
            assertEquals(mappingTable[floatIdx.getAndIncrement()], docId);
            return true;
        }).when(mockAcceptDocsForFloats).get(anyInt());

        bitsFromFloatVectors = floatVectorValues.getAcceptOrds(mockAcceptDocsForFloats);
        assertEquals(totalNumberOfVectors, bitsFromFloatVectors.length());
        for (int i = 0; i < totalNumberOfVectors; ++i) {
            bitsFromFloatVectors.get(i);
            assertEquals(mappingTable[i], floatVectorValues.ordToDoc(i));
        }
        assertEquals(totalNumberOfVectors, floatIdx.get());
    }

    public void testParentChildNestedCase() {
        doTestParentChildNestedCase(FaissIdMapIndex.IXMP);
        doTestParentChildNestedCase(FaissIdMapIndex.IBMP);
    }

    @SneakyThrows
    private void doTestParentChildNestedCase(final String indexType) {
        // Dimension : 128
        // #Vectors : 100
        // Metric : L2
        final int totalNumberOfVectors = 100;
        final int dimension = 128;
        final boolean l2Metric = true;

        // Prepare id mapping
        // Assuming each parent docs have 5 child docs.
        // e.g. [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, ...]
        final long[] mappingTable = new long[totalNumberOfVectors];
        for (int i = 0; i < totalNumberOfVectors; ++i) {
            mappingTable[i] = i / 5;
        }

        // Load index
        final FaissIdMapIndex index = triggerLoadAndGetIndex(dimension, totalNumberOfVectors, l2Metric, mappingTable, indexType);

        // We expect it to be null when identity case.
        final long[] loadedMappingTable = getVectorIdToDocIdMapping(index, totalNumberOfVectors);
        assertNotNull(loadedMappingTable);
        assertArrayEquals(mappingTable, loadedMappingTable);

        // Sparse byte vectors
        final ByteVectorValues byteVectorValues = index.getByteValues(null);
        Bits bitsFromByteVectors = byteVectorValues.getAcceptOrds(null);
        assertNull(bitsFromByteVectors);

        final Bits mockAcceptDocsForBytes = mock(Bits.class);
        final AtomicInteger byteIdx = new AtomicInteger(0);
        doAnswer((Answer<Boolean>) invocation -> {
            final int docId = invocation.getArgument(0);
            assertEquals(mappingTable[byteIdx.getAndIncrement()], docId);
            return true;
        }).when(mockAcceptDocsForBytes).get(anyInt());

        bitsFromByteVectors = byteVectorValues.getAcceptOrds(mockAcceptDocsForBytes);
        assertEquals(totalNumberOfVectors, bitsFromByteVectors.length());
        for (int i = 0; i < totalNumberOfVectors; ++i) {
            bitsFromByteVectors.get(i);
            assertEquals(mappingTable[i], byteVectorValues.ordToDoc(i));
        }
        assertEquals(totalNumberOfVectors, byteIdx.get());

        // Sparse float vectors
        final FloatVectorValues floatVectorValues = index.getFloatValues(null);
        Bits bitsFromFloatVectors = floatVectorValues.getAcceptOrds(null);
        assertNull(bitsFromFloatVectors);

        final Bits mockAcceptDocsForFloats = mock(Bits.class);
        final AtomicInteger floatIdx = new AtomicInteger(0);
        doAnswer((Answer<Boolean>) invocation -> {
            final int docId = invocation.getArgument(0);
            assertEquals(mappingTable[floatIdx.getAndIncrement()], docId);
            return true;
        }).when(mockAcceptDocsForFloats).get(anyInt());

        bitsFromFloatVectors = floatVectorValues.getAcceptOrds(mockAcceptDocsForFloats);
        assertEquals(totalNumberOfVectors, bitsFromFloatVectors.length());
        for (int i = 0; i < totalNumberOfVectors; ++i) {
            bitsFromFloatVectors.get(i);
            assertEquals(mappingTable[i], floatVectorValues.ordToDoc(i));
        }
        assertEquals(totalNumberOfVectors, floatIdx.get());
    }

    @SneakyThrows
    public void testLoadBinaryIdMapIndex() {
        final String relativePath = "data/memoryoptsearch/faiss_binary_50_vectors_512_dim.bin";
        final URL floatFloatVectors = FaissHNSWTests.class.getClassLoader().getResource(relativePath);
        byte[] bytes = Files.readAllBytes(Path.of(floatFloatVectors.toURI()));
        final IndexInput indexInput = new ByteArrayIndexInput("FaissIndexFloatFlatTests", bytes);

        final FaissIndex faissIndex = FaissIndex.load(indexInput);
        assert (faissIndex instanceof FaissIdMapIndex);

        final FaissIdMapIndex faissIdMapIndex = (FaissIdMapIndex) faissIndex;
        assertEquals(FaissIdMapIndex.IBMP, faissIdMapIndex.getIndexType());
        assertEquals(CODE_SIZE, faissIdMapIndex.getCodeSize());
    }

    @SneakyThrows
    private static long[] getVectorIdToDocIdMapping(final FaissIdMapIndex index, final int totalNumberOfVectors) {
        final Field field = FaissIdMapIndex.class.getDeclaredField("idMappingReader");
        field.setAccessible(true);
        DirectMonotonicReader decoder = (DirectMonotonicReader) field.get(index);
        if (decoder == null) {
            // It's an identical case
            return null;
        }
        long[] mappingTable = new long[totalNumberOfVectors];
        for (int i = 0; i < totalNumberOfVectors; ++i) {
            mappingTable[i] = decoder.get(i);
        }
        return mappingTable;
    }

    @SneakyThrows
    private static FaissIdMapIndex triggerLoadAndGetIndex(
        final int dimension,
        final long totalNumberOfVectors,
        final boolean useL2Metric,
        final long[] mappingTable,
        final String indexType
    ) {
        // Mock static `load` to return a dummy mock
        try (MockedStatic<FaissIndex> mockStaticFaissIndex = mockStatic(FaissIndex.class)) {
            // Nested index
            final FaissHNSWIndex nestedIndex = mock(FaissHNSWIndex.class);
            mockStaticFaissIndex.when(() -> FaissIndex.load(any())).thenReturn(nestedIndex);

            // Byte vectors
            final Bits mockBitsFromByteVectors = mock(Bits.class);
            final AtomicInteger idx1 = new AtomicInteger(0);
            doAnswer((Answer<Void>) invocation -> {
                final int convertedDocId = invocation.getArgument(0);
                assertEquals(mappingTable[idx1.getAndIncrement()], convertedDocId);
                return null;
            }).when(mockBitsFromByteVectors).get(anyInt());

            final ByteVectorValues mockByteValues = mock(FaissByteVectorValues.class);

            when(nestedIndex.getByteValues(any())).thenReturn(mockByteValues);
            when(mockByteValues.size()).thenReturn(Math.toIntExact(totalNumberOfVectors));

            // Float vectors
            final Bits mockBitsFromFloatVectors = mock(Bits.class);
            final AtomicInteger idx2 = new AtomicInteger(0);
            doAnswer((Answer<Void>) invocation -> {
                final int convertedDocId = invocation.getArgument(0);
                assertEquals(mappingTable[idx2.getAndIncrement()], convertedDocId);
                return null;
            }).when(mockBitsFromFloatVectors).get(anyInt());
            final FloatVectorValues mockFloatValues = mock(FaissFloatVectorValues.class);
            when(nestedIndex.getFloatValues(any())).thenReturn(mockFloatValues);
            when(mockFloatValues.size()).thenReturn(Math.toIntExact(totalNumberOfVectors));

            // Trigger load
            final IndexInput input = prepareBytes(dimension, totalNumberOfVectors, useL2Metric, mappingTable, indexType);
            final FaissIdMapIndex index = triggerDoLoad(input, indexType);
            return index;
        }
    }

    private static IndexInput prepareBytes(
        final int dimension,
        final long totalNumberOfVectors,
        final boolean useL2Metric,
        final long[] mappingTable,
        final String indexType
    ) {
        byte[] commonHeader;
        if (indexType.equals(FaissIdMapIndex.IXMP)) {
            commonHeader = FaissIndexTestUtils.makeCommonHeader(dimension, totalNumberOfVectors, useL2Metric);
        } else {
            commonHeader = FaissIndexTestUtils.makeBinaryCommonHeader(dimension, CODE_SIZE, totalNumberOfVectors);
        }
        final ByteBuffersDataOutput output = new ByteBuffersDataOutput();
        output.writeBytes(commonHeader);
        output.writeLong(mappingTable.length);
        for (final long mappedDocId : mappingTable) {
            output.writeLong(mappedDocId);
        }
        final byte[] bytes = output.toArrayCopy();
        return new ByteArrayIndexInput("FaissIdMapIndexTest", bytes);
    }

    @SneakyThrows
    private static FaissIdMapIndex triggerDoLoad(final IndexInput input, final String indexType) {
        final FaissIdMapIndex index = new FaissIdMapIndex(indexType);
        final Method doLoadMethod = FaissIdMapIndex.class.getDeclaredMethod("doLoad", IndexInput.class);
        doLoadMethod.setAccessible(true);
        doLoadMethod.invoke(index, input);
        return index;
    }

    private void validateHeader(final String indexType, final FaissIndex index, final int dimension, final long totalNumberOfVectors) {
        if (indexType.equals(FaissIdMapIndex.IXMP)) {
            assertEquals(dimension, index.getDimension());
            assertEquals(totalNumberOfVectors, index.getTotalNumberOfVectors());
            assertEquals(SpaceType.L2, index.getSpaceType());
        } else {
            final FaissBinaryIndex binaryIndex = (FaissBinaryIndex) index;
            assertEquals(dimension, binaryIndex.getDimension());
            assertEquals(CODE_SIZE, binaryIndex.getCodeSize());
            assertEquals(totalNumberOfVectors, binaryIndex.getTotalNumberOfVectors());
            assertEquals(SpaceType.HAMMING, binaryIndex.getSpaceType());
        }
    }
}
