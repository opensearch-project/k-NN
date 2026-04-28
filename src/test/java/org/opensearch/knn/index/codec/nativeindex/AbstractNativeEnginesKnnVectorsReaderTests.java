/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex;

import com.google.common.collect.ImmutableSet;
import lombok.SneakyThrows;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.IOSupplier;
import org.mockito.MockedStatic;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.codec.KNNCodecTestUtil;
import org.opensearch.knn.index.engine.BuiltinKNNEngine;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.memoryoptsearch.VectorSearcher;
import org.opensearch.knn.memoryoptsearch.VectorSearcherFactory;

import java.io.IOException;
import java.util.Collections;
import java.util.Set;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

public class AbstractNativeEnginesKnnVectorsReaderTests extends KNNTestCase {

    // --- checkIntegrity ---

    @SneakyThrows
    public void testCheckIntegrity_delegatesToFlatReader() {
        final FlatVectorsReader fvr = mock(FlatVectorsReader.class);
        createReader(new FieldInfos(new FieldInfo[0]), Collections.emptySet(), fvr).checkIntegrity();
        verify(fvr).checkIntegrity();
    }

    // --- getFloatVectorValues ---

    @SneakyThrows
    public void testGetFloatVectorValues_delegatesToFlatReader() {
        final FlatVectorsReader fvr = mock(FlatVectorsReader.class);
        final FloatVectorValues mockValues = mock(FloatVectorValues.class);
        when(fvr.getFloatVectorValues("f")).thenReturn(mockValues);
        assertSame(mockValues, createReader(new FieldInfos(new FieldInfo[0]), Collections.emptySet(), fvr).getFloatVectorValues("f"));
    }

    // --- getVectorSearcherSupplier ---

    @SneakyThrows
    public void testGetVectorSearcherSupplier_whenNonKnnField_thenReturnsNull() {
        final FieldInfo fi = KNNCodecTestUtil.FieldInfoBuilder.builder("field1").fieldNumber(0).build(); // no KNN_FIELD attribute
        final TestReader reader = createReader(
            new FieldInfos(new FieldInfo[] { fi }),
            Collections.emptySet(),
            mock(FlatVectorsReader.class)
        );
        assertNull(reader.getVectorSearcherSupplier(fi));
    }

    @SneakyThrows
    public void testGetVectorSearcherSupplier_whenNoEngine_thenReturnsNull() {
        // KNN_FIELD present but no KNN_ENGINE attribute → extractKNNEngine returns null
        final FieldInfo fi = KNNCodecTestUtil.FieldInfoBuilder.builder("field1")
            .fieldNumber(0)
            .addAttribute(KNNVectorFieldMapper.KNN_FIELD, "true")
            .build();
        final TestReader reader = createReader(
            new FieldInfos(new FieldInfo[] { fi }),
            Collections.emptySet(),
            mock(FlatVectorsReader.class)
        );
        assertNull(reader.getVectorSearcherSupplier(fi));
    }

    @SneakyThrows
    public void testGetVectorSearcherSupplier_whenNoSearcherFactory_thenReturnsNull() {
        final FieldInfo fi = createKnnFieldInfo("field1", BuiltinKNNEngine.FAISS, 0);
        KNNEngine mockFaiss = spy(BuiltinKNNEngine.FAISS);
        when(mockFaiss.getVectorSearcherFactory()).thenReturn(null);

        try (MockedStatic<BuiltinKNNEngine> ms = mockStatic(BuiltinKNNEngine.class)) {
            ms.when(() -> BuiltinKNNEngine.getEngine(any())).thenReturn(mockFaiss);
            ms.when(BuiltinKNNEngine::getEnginesThatCreateCustomSegmentFiles).thenReturn(ImmutableSet.of(mockFaiss));
            final TestReader reader = createReader(
                new FieldInfos(new FieldInfo[] { fi }),
                Collections.emptySet(),
                mock(FlatVectorsReader.class)
            );
            assertNull(reader.getVectorSearcherSupplier(fi));
        }
    }

    @SneakyThrows
    public void testGetVectorSearcherSupplier_whenNoSegmentFile_thenReturnsNull() {
        final FieldInfo fi = createKnnFieldInfo("field1", BuiltinKNNEngine.FAISS, 0);
        KNNEngine mockFaiss = spy(BuiltinKNNEngine.FAISS);
        when(mockFaiss.getVectorSearcherFactory()).thenReturn(mock(VectorSearcherFactory.class));

        try (MockedStatic<BuiltinKNNEngine> ms = mockStatic(BuiltinKNNEngine.class)) {
            ms.when(() -> BuiltinKNNEngine.getEngine(any())).thenReturn(mockFaiss);
            ms.when(BuiltinKNNEngine::getEnginesThatCreateCustomSegmentFiles).thenReturn(ImmutableSet.of(mockFaiss));
            // No files in segment → getNativeEngineFileFromFieldInfo returns null
            final TestReader reader = createReader(
                new FieldInfos(new FieldInfo[] { fi }),
                Collections.emptySet(),
                mock(FlatVectorsReader.class)
            );
            assertNull(reader.getVectorSearcherSupplier(fi));
        }
    }

    @SneakyThrows
    public void testGetVectorSearcherSupplier_whenAllConditionsMet_thenReturnsSupplier() {
        final FieldInfo fi = createKnnFieldInfo("field1", BuiltinKNNEngine.FAISS, 0);
        KNNEngine mockFaiss = spy(BuiltinKNNEngine.FAISS);
        VectorSearcherFactory mockFactory = mock(VectorSearcherFactory.class);
        when(mockFaiss.getVectorSearcherFactory()).thenReturn(mockFactory);
        when(mockFactory.createVectorSearcher(any(), anyString(), any(), any(), any())).thenReturn(mock(VectorSearcher.class));

        try (MockedStatic<BuiltinKNNEngine> ms = mockStatic(BuiltinKNNEngine.class)) {
            ms.when(() -> BuiltinKNNEngine.getEngine(any())).thenReturn(mockFaiss);
            ms.when(BuiltinKNNEngine::getEnginesThatCreateCustomSegmentFiles).thenReturn(ImmutableSet.of(mockFaiss));
            final TestReader reader = createReader(
                new FieldInfos(new FieldInfo[] { fi }),
                Set.of("_0_165_field1.faiss"),
                mock(FlatVectorsReader.class)
            );
            final IOSupplier<VectorSearcher> supplier = reader.getVectorSearcherSupplier(fi);
            assertNotNull(supplier);
            assertNotNull(supplier.get());
        }
    }

    // --- loadMemoryOptimizedSearcherIfRequired ---

    @SneakyThrows
    public void testLoadMemoryOptimizedSearcher_whenCalledTwice_thenSearcherCreatedOnce() {
        final FieldInfo fi = createKnnFieldInfo("field1", BuiltinKNNEngine.FAISS, 0);
        KNNEngine mockFaiss = spy(BuiltinKNNEngine.FAISS);
        VectorSearcherFactory mockFactory = mock(VectorSearcherFactory.class);
        VectorSearcher mockSearcher = mock(VectorSearcher.class);
        when(mockFaiss.getVectorSearcherFactory()).thenReturn(mockFactory);
        when(mockFactory.createVectorSearcher(any(), anyString(), any(), any(), any())).thenReturn(mockSearcher);

        try (MockedStatic<BuiltinKNNEngine> ms = mockStatic(BuiltinKNNEngine.class)) {
            ms.when(() -> BuiltinKNNEngine.getEngine(any())).thenReturn(mockFaiss);
            ms.when(BuiltinKNNEngine::getEnginesThatCreateCustomSegmentFiles).thenReturn(ImmutableSet.of(mockFaiss));
            final TestReader reader = createReader(
                new FieldInfos(new FieldInfo[] { fi }),
                Set.of("_0_165_field1.faiss"),
                mock(FlatVectorsReader.class)
            );
            final VectorSearcher first = reader.loadMemoryOptimizedSearcherIfRequired(fi);
            final VectorSearcher second = reader.loadMemoryOptimizedSearcherIfRequired(fi);
            assertSame(first, second);
            // factory called exactly once
            verify(mockFactory).createVectorSearcher(any(), anyString(), any(), any(), any());
        }
    }

    @SneakyThrows
    public void testLoadMemoryOptimizedSearcher_whenSupplierThrows_thenWrapsInRuntimeException() {
        final FieldInfo fi = createKnnFieldInfo("field1", BuiltinKNNEngine.FAISS, 0);
        KNNEngine mockFaiss = spy(BuiltinKNNEngine.FAISS);
        VectorSearcherFactory mockFactory = mock(VectorSearcherFactory.class);
        when(mockFaiss.getVectorSearcherFactory()).thenReturn(mockFactory);
        when(mockFactory.createVectorSearcher(any(), anyString(), any(), any(), any())).thenThrow(new IOException("disk error"));

        try (MockedStatic<BuiltinKNNEngine> ms = mockStatic(BuiltinKNNEngine.class)) {
            ms.when(() -> BuiltinKNNEngine.getEngine(any())).thenReturn(mockFaiss);
            ms.when(BuiltinKNNEngine::getEnginesThatCreateCustomSegmentFiles).thenReturn(ImmutableSet.of(mockFaiss));
            final TestReader reader = createReader(
                new FieldInfos(new FieldInfo[] { fi }),
                Set.of("_0_165_field1.faiss"),
                mock(FlatVectorsReader.class)
            );
            expectThrows(RuntimeException.class, () -> reader.loadMemoryOptimizedSearcherIfRequired(fi));
        }
    }

    // --- VectorSearcherHolder ---

    public void testVectorSearcherHolder_initiallyNotSet() {
        final AbstractNativeEnginesKnnVectorsReader.VectorSearcherHolder holder =
            new AbstractNativeEnginesKnnVectorsReader.VectorSearcherHolder();
        assertFalse(holder.isSet());
        assertNull(holder.getVectorSearcher());
    }

    public void testVectorSearcherHolder_afterSet_isSetReturnsTrue() {
        final AbstractNativeEnginesKnnVectorsReader.VectorSearcherHolder holder =
            new AbstractNativeEnginesKnnVectorsReader.VectorSearcherHolder();
        holder.setVectorSearcher(mock(VectorSearcher.class));
        assertTrue(holder.isSet());
        assertNotNull(holder.getVectorSearcher());
    }

    public void testVectorSearcherHolder_setWithNull_thenThrowsNPE() {
        final AbstractNativeEnginesKnnVectorsReader.VectorSearcherHolder holder =
            new AbstractNativeEnginesKnnVectorsReader.VectorSearcherHolder();
        expectThrows(NullPointerException.class, () -> holder.setVectorSearcher(null));
    }

    // --- helpers ---

    private static FieldInfo createKnnFieldInfo(String name, KNNEngine engine, int fieldNo) {
        return KNNCodecTestUtil.FieldInfoBuilder.builder(name)
            .fieldNumber(fieldNo)
            .addAttribute(KNNVectorFieldMapper.KNN_FIELD, "true")
            .addAttribute(KNNConstants.KNN_ENGINE, engine.getName())
            .build();
    }

    @SneakyThrows
    private static TestReader createReader(FieldInfos fieldInfos, Set<String> files, FlatVectorsReader fvr) {
        final Directory dir = mock(Directory.class);
        when(dir.openInput(any(), any())).thenReturn(mock(IndexInput.class));
        final SegmentInfo si = mock(SegmentInfo.class);
        when(si.files()).thenReturn(files);
        when(si.getId()).thenReturn((si.hashCode() + "").getBytes());
        return new TestReader(new SegmentReadState(dir, si, fieldInfos, IOContext.DEFAULT), fvr);
    }

    /** Minimal concrete subclass to test the abstract base class directly. */
    private static class TestReader extends AbstractNativeEnginesKnnVectorsReader {
        TestReader(SegmentReadState state, FlatVectorsReader fvr) {
            super(state, fvr);
        }

        @Override
        public void search(String field, float[] target, KnnCollector knnCollector, AcceptDocs acceptDocs) {}

        @Override
        public void search(String field, byte[] target, KnnCollector knnCollector, AcceptDocs acceptDocs) {}

        @Override
        public ByteVectorValues getByteVectorValues(String field) {
            return null;
        }

        @Override
        public void close() throws IOException {
            flatVectorsReader.close();
        }

        // Expose protected methods for testing
        @Override
        public IOSupplier<VectorSearcher> getVectorSearcherSupplier(FieldInfo fieldInfo) {
            return super.getVectorSearcherSupplier(fieldInfo);
        }

        @Override
        public VectorSearcher loadMemoryOptimizedSearcherIfRequired(FieldInfo fieldInfo) {
            return super.loadMemoryOptimizedSearcherIfRequired(fieldInfo);
        }

        @Override
        public void warmUp(String fieldName) throws IOException {}
    }
}
