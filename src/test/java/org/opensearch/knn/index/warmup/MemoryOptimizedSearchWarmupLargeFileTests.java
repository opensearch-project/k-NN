/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.warmup;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SerialMergeScheduler;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.tests.index.RandomIndexWriter;
import org.apache.lucene.tests.store.BaseDirectoryWrapper;
import org.mockito.MockedStatic;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.KNN990Codec.NativeEngines990KnnVectorsFormat;
import org.opensearch.knn.index.codec.util.UnitTestCodec;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.MemoryOptimizedSearchSupportSpec;
import org.opensearch.knn.index.mapper.KNNMappingConfig;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.index.mapper.KNNVectorFieldType;
import org.opensearch.knn.index.mapper.Mode;

import java.io.IOException;
import java.util.ArrayList;

import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.when;

/**
 * Non-parameterized tests for {@link MemoryOptimizedSearchWarmup} that test edge cases
 * like large file handling. These tests are separated from the parameterized tests
 * to avoid running expensive tests multiple times unnecessarily.
 */
public class MemoryOptimizedSearchWarmupLargeFileTests extends KNNTestCase {

    private static final Codec TESTING_CODEC = new UnitTestCodec(() -> new NativeEngines990KnnVectorsFormat(0));
    private static final String TEST_INDEX = "test-index";
    private static final String KNN_FIELD = "knn_field";

    private MemoryOptimizedSearchWarmup warmup;
    private Directory directory;
    private RandomIndexWriter indexWriter;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        warmup = new MemoryOptimizedSearchWarmup();
    }

    @Override
    public void tearDown() throws Exception {
        if (directory != null) {
            directory.close();
        }
        super.tearDown();
    }

    /**
     * Tests that warmup correctly handles large index files (> 2GB) without integer overflow.
     * This is a regression test for a bug where the loop counter was an int instead of long,
     * which would cause overflow when iterating over files larger than Integer.MAX_VALUE bytes.
     *
     * The fix was changing `for (int i = 0; ...)` to `for (long i = 0; ...)` in warmUpField().
     */
    @SneakyThrows
    public void testWarmUp_whenLargeIndexFile_thenNoIntegerOverflow() {
        // File size of 5GB to catch both signed and unsigned int overflow bugs
        // Integer.MAX_VALUE = 2,147,483,647 (~2.1GB) - signed int overflow
        // 2^32 = 4,294,967,296 (~4GB) - unsigned int overflow
        // 5GB ensures we catch both cases
        final long largeFileSize = 5L * 1024 * 1024 * 1024;

        // Use a lightweight IndexInput implementation instead of Mockito mocks
        // This avoids the overhead of mock invocation tracking
        FakeIndexInput fakeIndexInput = new FakeIndexInput(largeFileSize);

        // Create a fake directory that returns our fake IndexInput
        Directory fakeDirectory = new FakeDirectory(fakeIndexInput);

        // Set up a real index to get a valid LeafReader
        setupDirectory();
        addVectorDocument(KNN_FIELD);

        try (IndexReader reader = indexWriter.getReader()) {
            indexWriter.flush();
            indexWriter.commit();
            indexWriter.close();

            IndexSearcher searcher = new IndexSearcher(reader);
            LeafReader leafReader = searcher.getLeafContexts().get(0).reader();

            MapperService mapperService = mock(MapperService.class);
            KNNVectorFieldType knnFieldType = createMockedKnnFieldType();
            when(mapperService.fieldType(KNN_FIELD)).thenReturn(knnFieldType);

            try (MockedStatic<MemoryOptimizedSearchSupportSpec> supportSpecMock = mockStatic(MemoryOptimizedSearchSupportSpec.class)) {
                supportSpecMock.when(() -> MemoryOptimizedSearchSupportSpec.isSupportedFieldType(eq(knnFieldType), anyString()))
                    .thenReturn(true);

                // Use the fake directory instead of the real one
                ArrayList<String> result = warmup.warmUp(leafReader, mapperService, TEST_INDEX, fakeDirectory);

                // Verify warmup completed successfully
                assertEquals("Expected 1 warmed up field", 1, result.size());

                // Verify that we seeked past Integer.MAX_VALUE (proving no int overflow)
                assertTrue(
                    "Expected max seek position beyond Integer.MAX_VALUE, but was " + fakeIndexInput.maxSeekPosition,
                    fakeIndexInput.maxSeekPosition > Integer.MAX_VALUE
                );

                // Verify the last seek was to the end of the file (length - 1)
                assertEquals("Expected final seek to file end (length - 1)", largeFileSize - 1, fakeIndexInput.lastSeekPosition);
            }
        }
    }

    // ==================== Fake Implementations ====================

    /**
     * Lightweight IndexInput implementation for testing large file handling.
     * Avoids Mockito overhead by using simple field tracking.
     */
    private static class FakeIndexInput extends IndexInput {
        private final long length;
        long lastSeekPosition = -1;
        long maxSeekPosition = -1;

        FakeIndexInput(long length) {
            super("fake");
            this.length = length;
        }

        @Override
        public void close() {}

        @Override
        public long getFilePointer() {
            return lastSeekPosition;
        }

        @Override
        public void seek(long pos) {
            lastSeekPosition = pos;
            if (pos > maxSeekPosition) {
                maxSeekPosition = pos;
            }
        }

        @Override
        public long length() {
            return length;
        }

        @Override
        public IndexInput slice(String sliceDescription, long offset, long length) {
            return this;
        }

        @Override
        public byte readByte() {
            return 0;
        }

        @Override
        public void readBytes(byte[] b, int offset, int len) {}
    }

    /**
     * Minimal Directory implementation that returns a FakeIndexInput.
     */
    private static class FakeDirectory extends Directory {
        private final IndexInput indexInput;

        FakeDirectory(IndexInput indexInput) {
            this.indexInput = indexInput;
        }

        @Override
        public IndexInput openInput(String name, IOContext context) {
            return indexInput;
        }

        @Override
        public String[] listAll() {
            return new String[0];
        }

        @Override
        public void deleteFile(String name) {}

        @Override
        public long fileLength(String name) {
            return indexInput.length();
        }

        @Override
        public org.apache.lucene.store.IndexOutput createOutput(String name, IOContext context) {
            return null;
        }

        @Override
        public org.apache.lucene.store.IndexOutput createTempOutput(String prefix, String suffix, IOContext context) {
            return null;
        }

        @Override
        public void sync(java.util.Collection<String> names) {}

        @Override
        public void syncMetaData() {}

        @Override
        public void rename(String source, String dest) {}

        @Override
        public org.apache.lucene.store.Lock obtainLock(String name) {
            return null;
        }

        @Override
        public void close() {}

        @Override
        public java.util.Set<String> getPendingDeletions() {
            return java.util.Collections.emptySet();
        }
    }

    // ==================== Helper Methods ====================

    private void setupDirectory() throws IOException {
        directory = newFSDirectory(createTempDir());
        // Disable index checking on close since we're using native engine format
        ((BaseDirectoryWrapper) directory).setCheckIndexOnClose(false);
        indexWriter = createIndexWriter(directory);
    }

    private RandomIndexWriter createIndexWriter(Directory dir) throws IOException {
        IndexWriterConfig iwc = newIndexWriterConfig();
        iwc.setMergeScheduler(new SerialMergeScheduler());
        iwc.setCodec(TESTING_CODEC);
        iwc.setUseCompoundFile(false);
        iwc.setMergePolicy(NoMergePolicy.INSTANCE);
        return new RandomIndexWriter(random(), dir, iwc);
    }

    private KNNVectorFieldType createMockedKnnFieldType() {
        KNNVectorFieldType knnFieldType = mock(KNNVectorFieldType.class);
        KNNMappingConfig mappingConfig = mock(KNNMappingConfig.class);
        when(mappingConfig.getMode()).thenReturn(Mode.NOT_CONFIGURED);
        when(knnFieldType.getKnnMappingConfig()).thenReturn(mappingConfig);
        return knnFieldType;
    }

    private void addVectorDocument(String fieldName) throws IOException {
        Document doc = new Document();
        Field vectorField = createVectorField(fieldName);
        doc.add(vectorField);
        indexWriter.addDocument(doc);
    }

    private Field createVectorField(String fieldName) {
        int dimension = 3;
        FieldType fieldType = createVectorFieldType(dimension);
        float[] vector = generateFloatVector(dimension);
        return new KnnFloatVectorField(fieldName, vector, fieldType);
    }

    private float[] generateFloatVector(int dimension) {
        float[] vector = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            vector[i] = random().nextFloat() * 10 - 5;
        }
        return vector;
    }

    private FieldType createVectorFieldType(int dimension) {
        FieldType fieldType = new FieldType();
        fieldType.setTokenized(false);
        fieldType.setIndexOptions(IndexOptions.NONE);
        fieldType.putAttribute(KNNVectorFieldMapper.KNN_FIELD, "true");
        fieldType.putAttribute(KNNConstants.KNN_METHOD, KNNConstants.METHOD_HNSW);
        fieldType.putAttribute(KNNConstants.KNN_ENGINE, KNNEngine.FAISS.getName());
        fieldType.putAttribute(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue());
        fieldType.putAttribute(KNNConstants.HNSW_ALGO_M, "16");
        fieldType.putAttribute(KNNConstants.HNSW_ALGO_EF_CONSTRUCTION, "100");
        fieldType.putAttribute(KNNConstants.VECTOR_DATA_TYPE_FIELD, VectorDataType.FLOAT.getValue());
        fieldType.putAttribute(KNNConstants.PARAMETERS, "{ \"index_description\":\"HNSW16,Flat\", \"spaceType\": \"l2\"}");
        fieldType.setVectorAttributes(
            dimension,
            VectorEncoding.FLOAT32,
            SpaceType.L2.getKnnVectorSimilarityFunction().getVectorSimilarityFunction()
        );
        fieldType.freeze();
        return fieldType;
    }
}
