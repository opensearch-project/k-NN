/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import com.google.common.collect.ImmutableSet;
import lombok.SneakyThrows;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.mockito.MockedStatic;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.codec.KNNCodecTestUtil;
import org.opensearch.knn.index.codec.nativeindex.AbstractNativeEnginesKnnVectorsReader;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.memoryoptsearch.VectorSearcher;
import org.opensearch.knn.memoryoptsearch.VectorSearcherFactory;

import java.lang.reflect.Field;
import java.util.Collections;
import java.util.Set;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

public class Faiss1040ScalarQuantizedKnnVectorsReaderTests extends KNNTestCase {

    @SneakyThrows
    public void testCheckIntegrity_thenDelegatesToFlatReader() {
        final FlatVectorsReader fvr = mock(FlatVectorsReader.class);
        createReader(new FieldInfos(new FieldInfo[0]), Collections.emptySet(), fvr).checkIntegrity();
        verify(fvr).checkIntegrity();
    }

    @SneakyThrows
    public void testGetFloatVectorValues_thenDelegatesToFlatReader() {
        final FlatVectorsReader fvr = mock(FlatVectorsReader.class);
        final FloatVectorValues mockValues = mock(FloatVectorValues.class);
        when(fvr.getFloatVectorValues("f")).thenReturn(mockValues);
        assertSame(mockValues, createReader(new FieldInfos(new FieldInfo[0]), Collections.emptySet(), fvr).getFloatVectorValues("f"));
    }

    @SneakyThrows
    public void testGetByteVectorValues_thenThrowsUnsupported() {
        expectThrows(
            UnsupportedOperationException.class,
            () -> createReader(new FieldInfos(new FieldInfo[0]), Collections.emptySet(), mock(FlatVectorsReader.class)).getByteVectorValues(
                "f"
            )
        );
    }

    @SneakyThrows
    public void testSearchFloat_whenNullTarget_thenThrowsException() {
        final FieldInfo fi = createFieldInfo("field1", KNNEngine.FAISS, 0);
        KNNEngine mockFaiss = spy(KNNEngine.FAISS);
        VectorSearcherFactory mockFactory = mock(VectorSearcherFactory.class);
        when(mockFaiss.getVectorSearcherFactory()).thenReturn(mockFactory);
        when(
            mockFactory.createVectorSearcher(
                any(Directory.class),
                anyString(),
                any(FieldInfo.class),
                any(IOContext.class),
                any(FlatVectorsReader.class)
            )
        ).thenReturn(mock(VectorSearcher.class));

        try (MockedStatic<KNNEngine> ms = mockStatic(KNNEngine.class)) {
            ms.when(() -> KNNEngine.getEngine(any())).thenReturn(mockFaiss);
            ms.when(KNNEngine::getEnginesThatCreateCustomSegmentFiles).thenReturn(ImmutableSet.of(mockFaiss));
            final Faiss1040ScalarQuantizedKnnVectorsReader reader = createReader(
                new FieldInfos(new FieldInfo[] { fi }),
                Set.of("_0_165_field1.faiss"),
                mock(FlatVectorsReader.class)
            );
            expectThrows(Exception.class, () -> reader.search("field1", (float[]) null, null, null));
        }
    }

    @SneakyThrows
    public void testSearchFloat_whenSearcherAvailable_thenDelegates() {
        final FieldInfo fi = createFieldInfo("field1", KNNEngine.FAISS, 0);
        KNNEngine mockFaiss = spy(KNNEngine.FAISS);
        VectorSearcherFactory mockFactory = mock(VectorSearcherFactory.class);
        VectorSearcher mockSearcher = mock(VectorSearcher.class);
        when(mockFaiss.getVectorSearcherFactory()).thenReturn(mockFactory);
        when(
            mockFactory.createVectorSearcher(
                any(Directory.class),
                anyString(),
                any(FieldInfo.class),
                any(IOContext.class),
                any(FlatVectorsReader.class)
            )
        ).thenReturn(mockSearcher);
        try (MockedStatic<KNNEngine> ms = mockStatic(KNNEngine.class)) {
            ms.when(() -> KNNEngine.getEngine(any())).thenReturn(mockFaiss);
            ms.when(KNNEngine::getEnginesThatCreateCustomSegmentFiles).thenReturn(ImmutableSet.of(mockFaiss));
            final Faiss1040ScalarQuantizedKnnVectorsReader reader = createReader(
                new FieldInfos(new FieldInfo[] { fi }),
                Set.of("_0_165_field1.faiss"),
                mock(FlatVectorsReader.class)
            );
            float[] target = { 1, 2, 3 };
            reader.search("field1", target, null, null);
            verify(mockSearcher).search(target, null, null);
        }
    }

    @SneakyThrows
    public void testSearchFloat_whenNoSearcher_thenThrowsIllegalState() {
        final FieldInfo fi = createFieldInfo("field1", KNNEngine.FAISS, 0);
        KNNEngine mockFaiss = spy(KNNEngine.FAISS);
        when(mockFaiss.getVectorSearcherFactory()).thenReturn(null);

        try (MockedStatic<KNNEngine> ms = mockStatic(KNNEngine.class)) {
            ms.when(() -> KNNEngine.getEngine(any())).thenReturn(mockFaiss);
            ms.when(KNNEngine::getEnginesThatCreateCustomSegmentFiles).thenReturn(ImmutableSet.of(mockFaiss));
            final Faiss1040ScalarQuantizedKnnVectorsReader reader = createReader(
                new FieldInfos(new FieldInfo[] { fi }),
                Collections.emptySet(),
                mock(FlatVectorsReader.class)
            );
            expectThrows(IllegalStateException.class, () -> reader.search("field1", new float[] { 1, 2, 3 }, null, null));
        }
    }

    @SneakyThrows
    public void testSearchByte_thenThrowsUnsupported() {
        expectThrows(
            UnsupportedOperationException.class,
            () -> createReader(new FieldInfos(new FieldInfo[0]), Collections.emptySet(), mock(FlatVectorsReader.class)).search(
                "f",
                new byte[] { 1 },
                null,
                null
            )
        );
    }

    @SneakyThrows
    public void testClose_thenClosesFlatReaderAndSearcher() {
        final FieldInfo fi = createFieldInfo("field1", KNNEngine.FAISS, 0);
        KNNEngine mockFaiss = spy(KNNEngine.FAISS);
        VectorSearcherFactory mockFactory = mock(VectorSearcherFactory.class);
        VectorSearcher mockSearcher = mock(VectorSearcher.class);
        when(mockFaiss.getVectorSearcherFactory()).thenReturn(mockFactory);
        when(
            mockFactory.createVectorSearcher(
                any(Directory.class),
                anyString(),
                any(FieldInfo.class),
                any(IOContext.class),
                any(FlatVectorsReader.class)
            )
        ).thenReturn(mockSearcher);
        final FlatVectorsReader fvr = mock(FlatVectorsReader.class);

        try (MockedStatic<KNNEngine> ms = mockStatic(KNNEngine.class)) {
            ms.when(() -> KNNEngine.getEngine(any())).thenReturn(mockFaiss);
            ms.when(KNNEngine::getEnginesThatCreateCustomSegmentFiles).thenReturn(ImmutableSet.of(mockFaiss));
            final Faiss1040ScalarQuantizedKnnVectorsReader reader = createReader(
                new FieldInfos(new FieldInfo[] { fi }),
                Set.of("_0_165_field1.faiss"),
                fvr
            );
            reader.search("field1", new float[] { 1, 2, 3 }, null, null);
            reader.close();
            verify(fvr).close();
            verify(mockSearcher).close();
        }
    }

    @SneakyThrows
    public void testClose_whenNoSearcher_thenClosesFlatReaderOnly() {
        final FlatVectorsReader fvr = mock(FlatVectorsReader.class);
        createReader(new FieldInfos(new FieldInfo[0]), Collections.emptySet(), fvr).close();
        verify(fvr).close();
    }

    @SneakyThrows
    public void testVectorSearcherHolder_initiallyNotSet() {
        final Faiss1040ScalarQuantizedKnnVectorsReader reader = createReader(
            new FieldInfos(new FieldInfo[0]),
            Collections.emptySet(),
            mock(FlatVectorsReader.class)
        );
        final Field f = AbstractNativeEnginesKnnVectorsReader.class.getDeclaredField("vectorSearcherHolder");
        f.setAccessible(true);
        assertFalse(((AbstractNativeEnginesKnnVectorsReader.VectorSearcherHolder) f.get(reader)).isSet());
    }

    // --- helpers ---

    private static FieldInfo createFieldInfo(String name, KNNEngine engine, int fieldNo) {
        KNNCodecTestUtil.FieldInfoBuilder b = KNNCodecTestUtil.FieldInfoBuilder.builder(name).fieldNumber(fieldNo);
        if (engine != null) {
            b.addAttribute(KNNVectorFieldMapper.KNN_FIELD, "true").addAttribute(KNNConstants.KNN_ENGINE, engine.getName());
        }
        return b.build();
    }

    @SneakyThrows
    private static Faiss1040ScalarQuantizedKnnVectorsReader createReader(FieldInfos fieldInfos, Set<String> files, FlatVectorsReader fvr) {
        Directory dir = mock(Directory.class);
        when(dir.openInput(any(), any())).thenReturn(mock(IndexInput.class));
        SegmentInfo si = mock(SegmentInfo.class);
        when(si.files()).thenReturn(files);
        when(si.getId()).thenReturn((si.hashCode() + "").getBytes());
        return new Faiss1040ScalarQuantizedKnnVectorsReader(new SegmentReadState(dir, si, fieldInfos, IOContext.DEFAULT), fvr);
    }
}
