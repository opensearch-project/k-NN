/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.hnsw.FlatVectorScorerUtil;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorScorer;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.mockito.MockedStatic;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.memoryoptsearch.faiss.FaissHNSW;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIdMapIndex;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndex;
import org.opensearch.knn.memoryoptsearch.faiss.FaissMemoryOptimizedSearcherFactory;
import org.opensearch.knn.memoryoptsearch.faiss.FaissScalarQuantizedFlatIndex;
import org.opensearch.knn.memoryoptsearch.faiss.UnsupportedFaissIndexException;
import org.opensearch.knn.memoryoptsearch.faiss.binary.FaissBinaryHnswIndex;

import java.io.IOException;
import java.lang.reflect.Field;
import java.nio.file.Path;
import java.util.UUID;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.when;

public class FaissMemoryOptimizedSearcherFactoryTests extends KNNTestCase {

    private static final FlatVectorsScorer LUCENE99_SCORER = FlatVectorScorerUtil.getLucene99FlatVectorsScorer();
    private static final FlatVectorsScorer SQ_SCORER = new Lucene104ScalarQuantizedVectorScorer(LUCENE99_SCORER);
    private static final FlatVectorsScorer SCORER = LUCENE99_SCORER;

    @SneakyThrows
    public void testCreateVectorSearcher_whenScalarQuantizedFieldWith1Bit_thenWiresFlatIndexAndScorer() {
        final FaissMemoryOptimizedSearcherFactory factory = new FaissMemoryOptimizedSearcherFactory();
        final Path tempDir = createTempDir(UUID.randomUUID().toString());
        final String fileName = "_0_test_field.faiss";

        // Build a mock index tree: FaissIdMapIndex -> FaissHNSWIndex -> FaissEmptyIndex
        // so that maybeSetFlatIndex replaces the empty storage with FaissScalarQuantizedFlatIndex
        final FaissHNSW faissHNSW = mock(FaissHNSW.class);
        final FaissBinaryHnswIndex hnswIndex = new FaissBinaryHnswIndex(FaissBinaryHnswIndex.IBHF, faissHNSW);
        hnswIndex.setStorage(null);

        // Set nested index
        final FaissIdMapIndex idMapIndex = new FaissIdMapIndex(FaissIdMapIndex.IXMP);
        final Field nestedIndexField = FaissIdMapIndex.class.getDeclaredField("nestedIndex");
        nestedIndexField.setAccessible(true);
        nestedIndexField.set(idMapIndex, hnswIndex);

        // Set Hnsw index
        final Field hnswGetterField = FaissIdMapIndex.class.getDeclaredField("hnswGetter");
        hnswGetterField.setAccessible(true);
        hnswGetterField.set(idMapIndex, hnswIndex);

        // Set space type
        final Field spaceTypeField = FaissIndex.class.getDeclaredField("spaceType");
        spaceTypeField.setAccessible(true);
        spaceTypeField.set(idMapIndex, SpaceType.L2);

        // Setting field info
        final FieldInfo fieldInfo = mock(FieldInfo.class);
        when(fieldInfo.getAttribute(KNNConstants.SQ_CONFIG)).thenReturn("bits=1");
        when(fieldInfo.getAttribute(KNNConstants.SPACE_TYPE)).thenReturn(SpaceType.L2.getValue());
        when(fieldInfo.getName()).thenReturn("test_field");

        // Set flat vector scorer
        final FlatVectorsReader flatVectorsReader = mock(FlatVectorsReader.class);
        when(flatVectorsReader.getFlatVectorScorer()).thenReturn(SQ_SCORER);

        try (Directory directory = newFSDirectory(tempDir)) {
            try (IndexOutput output = directory.createOutput(fileName, IOContext.DEFAULT)) {
                output.writeBytes(new byte[16], 16); // placeholder bytes, load is mocked
            }

            try (MockedStatic<FaissIndex> mockStaticFaissIndex = mockStatic(FaissIndex.class)) {
                mockStaticFaissIndex.when(() -> FaissIndex.load(any(IndexInput.class))).thenReturn(idMapIndex);

                final VectorSearcher searcher = factory.createVectorSearcher(
                    directory,
                    fileName,
                    fieldInfo,
                    IOContext.DEFAULT,
                    flatVectorsReader
                );

                assertNotNull(searcher);
                // Verify flat storage was wired in
                assertTrue(hnswIndex.getStorage() instanceof FaissScalarQuantizedFlatIndex);
                searcher.close();
            }
        }
    }

    @SneakyThrows
    public void testCreateVectorSearcher_whenValidFaissIndex_thenReturnsSearcher() {
        final FaissMemoryOptimizedSearcherFactory factory = new FaissMemoryOptimizedSearcherFactory();
        final Path tempDir = createTempDir(UUID.randomUUID().toString());
        final String fileName = "test_index.faiss";

        // Use a valid FAISS HNSW index binary (flat index types are not supported by the searcher)
        final byte[] indexBytes = loadResourceBytes("data/memoryoptsearch/faiss_cagra_flat_float_300_vectors_768_dims.bin");

        try (Directory directory = newFSDirectory(tempDir)) {
            try (IndexOutput output = directory.createOutput(fileName, IOContext.DEFAULT)) {
                output.writeBytes(indexBytes, indexBytes.length);
            }

            FieldInfo fieldInfo = mock(FieldInfo.class);
            when(fieldInfo.getAttribute(KNNConstants.SPACE_TYPE)).thenReturn(SpaceType.L2.getValue());

            FlatVectorsReader flatVectorsReader = mock(FlatVectorsReader.class);
            when(flatVectorsReader.getFlatVectorScorer()).thenReturn(SCORER);

            VectorSearcher searcher = factory.createVectorSearcher(directory, fileName, fieldInfo, IOContext.DEFAULT, flatVectorsReader);
            assertNotNull(searcher);
            searcher.close();
        }
    }

    @SneakyThrows
    public void testCreateVectorSearcher_whenUnsupportedIndex_thenThrowsAndClosesInput() {
        final FaissMemoryOptimizedSearcherFactory factory = new FaissMemoryOptimizedSearcherFactory();
        final Path tempDir = createTempDir(UUID.randomUUID().toString());
        final String fileName = "invalid_index.faiss";

        // Write garbage bytes that will fail FaissIndex.load with UnsupportedFaissIndexException
        try (Directory directory = newFSDirectory(tempDir)) {
            try (IndexOutput output = directory.createOutput(fileName, IOContext.DEFAULT)) {
                // Write an invalid FAISS index type header to trigger UnsupportedFaissIndexException
                // FAISS index format: first 4 bytes are a uint32 representing the index type string length,
                // followed by the index type string. We write a valid-looking but unsupported type.
                byte[] unsupportedType = "IxUNSUPPORTED".getBytes();
                output.writeInt(unsupportedType.length);
                output.writeBytes(unsupportedType, unsupportedType.length);
                // Pad with zeros to avoid EOF
                output.writeBytes(new byte[1024], 1024);
            }

            FieldInfo fieldInfo = mock(FieldInfo.class);
            when(fieldInfo.getAttribute(KNNConstants.SPACE_TYPE)).thenReturn(SpaceType.L2.getValue());

            FlatVectorsReader flatVectorsReader = mock(FlatVectorsReader.class);
            when(flatVectorsReader.getFlatVectorScorer()).thenReturn(SCORER);

            expectThrows(
                UnsupportedFaissIndexException.class,
                () -> factory.createVectorSearcher(directory, fileName, fieldInfo, IOContext.DEFAULT, flatVectorsReader)
            );
        }
    }

    @SneakyThrows
    public void testCreateVectorSearcher_whenIOExceptionOnOpen_thenThrowsIOException() {
        final FaissMemoryOptimizedSearcherFactory factory = new FaissMemoryOptimizedSearcherFactory();
        final Path tempDir = createTempDir(UUID.randomUUID().toString());

        try (Directory directory = newFSDirectory(tempDir)) {
            FieldInfo fieldInfo = mock(FieldInfo.class);
            FlatVectorsReader flatVectorsReader = mock(FlatVectorsReader.class);
            when(flatVectorsReader.getFlatVectorScorer()).thenReturn(SCORER);

            // File doesn't exist, should throw IOException
            expectThrows(
                IOException.class,
                () -> factory.createVectorSearcher(directory, "nonexistent.faiss", fieldInfo, IOContext.DEFAULT, flatVectorsReader)
            );
        }
    }

    @SneakyThrows
    public void testCreateVectorSearcher_whenUnsupportedIndex_thenIndexInputIsClosed() {
        final FaissMemoryOptimizedSearcherFactory factory = new FaissMemoryOptimizedSearcherFactory();
        final Path tempDir = createTempDir(UUID.randomUUID().toString());
        final String fileName = "invalid_index2.faiss";

        try (Directory directory = newFSDirectory(tempDir)) {
            try (IndexOutput output = directory.createOutput(fileName, IOContext.DEFAULT)) {
                byte[] unsupportedType = "IxUNSUPPORTED".getBytes();
                output.writeInt(unsupportedType.length);
                output.writeBytes(unsupportedType, unsupportedType.length);
                output.writeBytes(new byte[1024], 1024);
            }

            FieldInfo fieldInfo = mock(FieldInfo.class);
            when(fieldInfo.getAttribute(KNNConstants.SPACE_TYPE)).thenReturn(SpaceType.L2.getValue());

            FlatVectorsReader flatVectorsReader = mock(FlatVectorsReader.class);
            when(flatVectorsReader.getFlatVectorScorer()).thenReturn(SCORER);

            // Verify that after the exception, the IndexInput was properly closed
            // by confirming we can open the file again (no resource leak)
            try {
                factory.createVectorSearcher(directory, fileName, fieldInfo, IOContext.DEFAULT, flatVectorsReader);
                fail("Expected UnsupportedFaissIndexException");
            } catch (UnsupportedFaissIndexException e) {
                // Expected - now verify we can still open the file (no leaked handles)
                IndexInput input = directory.openInput(fileName, IOContext.DEFAULT);
                assertNotNull(input);
                input.close();
            }
        }
    }

    @SneakyThrows
    private byte[] loadResourceBytes(String resourcePath) {
        return FaissHNSWTests.class.getClassLoader().getResourceAsStream(resourcePath).readAllBytes();
    }
}
