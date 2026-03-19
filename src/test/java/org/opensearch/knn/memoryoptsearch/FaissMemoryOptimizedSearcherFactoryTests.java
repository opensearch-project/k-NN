/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.hnsw.FlatVectorScorerUtil;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.memoryoptsearch.faiss.FaissMemoryOptimizedSearcherFactory;
import org.opensearch.knn.memoryoptsearch.faiss.UnsupportedFaissIndexException;

import java.io.IOException;
import java.nio.file.Path;
import java.util.UUID;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class FaissMemoryOptimizedSearcherFactoryTests extends KNNTestCase {

    private static final FlatVectorsScorer SCORER = FlatVectorScorerUtil.getLucene99FlatVectorsScorer();

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
