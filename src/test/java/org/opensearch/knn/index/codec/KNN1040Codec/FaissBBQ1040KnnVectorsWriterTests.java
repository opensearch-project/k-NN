/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.util.Version;
import org.mockito.Mock;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.KNN990Codec.NativeEngineFieldVectorsWriter;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexWriter;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;
import org.opensearch.knn.index.vectorvalues.TestVectorValues;
import org.opensearch.knn.plugin.stats.KNNGraphValue;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

public class FaissBBQ1040KnnVectorsWriterTests extends KNNTestCase {

    @Mock
    private FlatVectorsWriter flatVectorsWriter;
    @Mock
    private SegmentWriteState segmentWriteState;
    @Mock
    private NativeIndexWriter nativeIndexWriter;
    @Mock
    private SegmentInfo segmentInfo;

    private FlatFieldVectorsWriter mockedFlatFieldVectorsWriter;
    private FaissBBQ1040KnnVectorsWriter objectUnderTest;

    private static final int BUILD_GRAPH_ALWAYS_THRESHOLD = 0;
    private static final int BUILD_GRAPH_NEVER_THRESHOLD = -1;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        MockitoAnnotations.openMocks(this);
        objectUnderTest = new FaissBBQ1040KnnVectorsWriter(segmentWriteState, flatVectorsWriter, BUILD_GRAPH_ALWAYS_THRESHOLD);
        mockedFlatFieldVectorsWriter = mock(FlatFieldVectorsWriter.class);
        Mockito.doNothing().when(mockedFlatFieldVectorsWriter).addValue(Mockito.anyInt(), Mockito.any());
        when(flatVectorsWriter.addField(any())).thenReturn(mockedFlatFieldVectorsWriter);

        try {
            Field infoField = SegmentWriteState.class.getDeclaredField("segmentInfo");
            infoField.setAccessible(true);
            infoField.set(segmentWriteState, segmentInfo);
        } catch (Exception ignored) {}
        when(segmentInfo.getVersion()).thenReturn(Version.LATEST);
    }

    // --- addField ---

    @SneakyThrows
    public void testAddField_thenSuccess() {
        final FieldInfo fieldInfo = fieldInfo(0, VectorEncoding.FLOAT32, Map.of(KNNConstants.VECTOR_DATA_TYPE_FIELD, "float"));
        NativeEngineFieldVectorsWriter field = nativeEngineFieldVectorsWriter(fieldInfo, Map.of(0, new float[] { 1, 2, 3 }));
        try (MockedStatic<NativeEngineFieldVectorsWriter> ms = mockStatic(NativeEngineFieldVectorsWriter.class)) {
            ms.when(() -> NativeEngineFieldVectorsWriter.create(fieldInfo, mockedFlatFieldVectorsWriter, segmentWriteState.infoStream))
                .thenReturn(field);
            objectUnderTest.addField(fieldInfo);
            verify(flatVectorsWriter).addField(fieldInfo);
        }
    }

    @SneakyThrows
    public void testAddField_calledTwice_thenThrowsException() {
        final FieldInfo fi1 = fieldInfo(0, VectorEncoding.FLOAT32, Map.of(KNNConstants.VECTOR_DATA_TYPE_FIELD, "float"));
        final FieldInfo fi2 = fieldInfo(1, VectorEncoding.FLOAT32, Map.of(KNNConstants.VECTOR_DATA_TYPE_FIELD, "float"));
        NativeEngineFieldVectorsWriter field = nativeEngineFieldVectorsWriter(fi1, Map.of(0, new float[] { 1, 2, 3 }));
        try (MockedStatic<NativeEngineFieldVectorsWriter> ms = mockStatic(NativeEngineFieldVectorsWriter.class)) {
            ms.when(() -> NativeEngineFieldVectorsWriter.create(fi1, mockedFlatFieldVectorsWriter, segmentWriteState.infoStream))
                .thenReturn(field);
            objectUnderTest.addField(fi1);
            expectThrows(IllegalStateException.class, () -> objectUnderTest.addField(fi2));
        }
    }

    // --- flush ---

    @SneakyThrows
    public void testFlush_whenNoField_thenOnlyDelegates() {
        objectUnderTest.flush(5, null);
        verify(flatVectorsWriter).flush(5, null);
    }

    @SneakyThrows
    public void testFlush_whenNoLiveDocs_thenSkipsBuild() {
        final FieldInfo fi = fieldInfo(0, VectorEncoding.FLOAT32, Map.of(KNNConstants.VECTOR_DATA_TYPE_FIELD, "float"));
        NativeEngineFieldVectorsWriter field = nativeEngineFieldVectorsWriter(fi, Map.of());
        try (MockedStatic<NativeEngineFieldVectorsWriter> ms = mockStatic(NativeEngineFieldVectorsWriter.class)) {
            ms.when(() -> NativeEngineFieldVectorsWriter.create(fi, mockedFlatFieldVectorsWriter, segmentWriteState.infoStream))
                .thenReturn(field);
            objectUnderTest.addField(fi);
            objectUnderTest.flush(5, null);
            verify(flatVectorsWriter).flush(5, null);
        }
    }

    @SneakyThrows
    public void testFlush_whenBelowThreshold_thenSkipsBuild() {
        FaissBBQ1040KnnVectorsWriter w = new FaissBBQ1040KnnVectorsWriter(segmentWriteState, flatVectorsWriter, 100);
        final FieldInfo fi = fieldInfo(0, VectorEncoding.FLOAT32, Map.of(KNNConstants.VECTOR_DATA_TYPE_FIELD, "float"));
        NativeEngineFieldVectorsWriter field = nativeEngineFieldVectorsWriter(fi, Map.of(0, new float[] { 1, 2, 3 }));
        try (MockedStatic<NativeEngineFieldVectorsWriter> ms = mockStatic(NativeEngineFieldVectorsWriter.class)) {
            ms.when(() -> NativeEngineFieldVectorsWriter.create(fi, mockedFlatFieldVectorsWriter, segmentWriteState.infoStream))
                .thenReturn(field);
            w.addField(fi);
            w.flush(5, null);
            verify(flatVectorsWriter).flush(5, null);
        }
    }

    @SneakyThrows
    public void testFlush_whenNegativeThreshold_thenSkipsBuild() {
        FaissBBQ1040KnnVectorsWriter w = new FaissBBQ1040KnnVectorsWriter(
            segmentWriteState,
            flatVectorsWriter,
            BUILD_GRAPH_NEVER_THRESHOLD
        );
        final FieldInfo fi = fieldInfo(0, VectorEncoding.FLOAT32, Map.of(KNNConstants.VECTOR_DATA_TYPE_FIELD, "float"));
        NativeEngineFieldVectorsWriter field = nativeEngineFieldVectorsWriter(fi, Map.of(0, new float[] { 1, 2, 3 }));
        try (MockedStatic<NativeEngineFieldVectorsWriter> ms = mockStatic(NativeEngineFieldVectorsWriter.class)) {
            ms.when(() -> NativeEngineFieldVectorsWriter.create(fi, mockedFlatFieldVectorsWriter, segmentWriteState.infoStream))
                .thenReturn(field);
            w.addField(fi);
            w.flush(5, null);
            verify(flatVectorsWriter).flush(5, null);
        }
    }

    @SneakyThrows
    public void testFlush_whenAboveThreshold_thenBuildsIndex() {
        final FieldInfo fi = fieldInfo(
            0,
            VectorEncoding.FLOAT32,
            Map.of(KNNConstants.VECTOR_DATA_TYPE_FIELD, "float", KNNConstants.KNN_ENGINE, "faiss")
        );
        Map<Integer, float[]> vectors = Map.of(0, new float[] { 1, 2, 3 }, 1, new float[] { 4, 5, 6 });
        final TestVectorValues.PreDefinedFloatVectorValues pv = new TestVectorValues.PreDefinedFloatVectorValues(
            new ArrayList<>(vectors.values())
        );
        final Supplier<KNNVectorValues<?>> expectedSupplier = KNNVectorValuesFactory.getVectorValuesSupplier(VectorDataType.FLOAT, pv);
        NativeEngineFieldVectorsWriter field = nativeEngineFieldVectorsWriter(fi, vectors);

        try (
            MockedStatic<NativeEngineFieldVectorsWriter> ms = mockStatic(NativeEngineFieldVectorsWriter.class);
            MockedStatic<KNNVectorValuesFactory> kvf = mockStatic(KNNVectorValuesFactory.class);
            MockedStatic<NativeIndexWriter> niw = mockStatic(NativeIndexWriter.class)
        ) {
            ms.when(() -> NativeEngineFieldVectorsWriter.create(fi, mockedFlatFieldVectorsWriter, segmentWriteState.infoStream))
                .thenReturn(field);
            objectUnderTest.addField(fi);

            kvf.when(
                () -> KNNVectorValuesFactory.getVectorValuesSupplier(
                    VectorDataType.FLOAT,
                    field.getFlatFieldVectorsWriter().getDocsWithFieldSet(),
                    vectors
                )
            ).thenReturn(expectedSupplier);
            niw.when(() -> NativeIndexWriter.getWriter(any(FieldInfo.class), any(SegmentWriteState.class))).thenReturn(nativeIndexWriter);
            doAnswer(a -> {
                Thread.sleep(2);
                return null;
            }).when(nativeIndexWriter).flushIndex(any(), anyInt());

            objectUnderTest.flush(5, null);

            verify(flatVectorsWriter).flush(5, null);
            verify(nativeIndexWriter).flushIndex(expectedSupplier, vectors.size());
            assertNotEquals(0L, (long) KNNGraphValue.REFRESH_TOTAL_TIME_IN_MILLIS.getValue());
        }
    }

    // --- mergeOneField ---

    @SneakyThrows
    public void testMergeOneField_whenNoLiveDocs_thenSkipsBuild() {
        final FieldInfo fi = fieldInfo(
            0,
            VectorEncoding.FLOAT32,
            Map.of(KNNConstants.VECTOR_DATA_TYPE_FIELD, "float", KNNConstants.KNN_ENGINE, "faiss")
        );
        final KNNVectorValues<float[]> kv = KNNVectorValuesFactory.getVectorValues(
            VectorDataType.FLOAT,
            new TestVectorValues.PreDefinedFloatVectorValues(new ArrayList<>())
        );
        final Supplier<KNNVectorValues<?>> supplier = () -> kv;

        try (MockedStatic<KNNVectorValuesFactory> kvf = mockStatic(KNNVectorValuesFactory.class)) {
            kvf.when(() -> KNNVectorValuesFactory.getKNNVectorValuesSupplierForMerge(any(), any(), any())).thenReturn(supplier);
            objectUnderTest.mergeOneField(fi, mock(MergeState.class));
            verify(flatVectorsWriter).mergeOneField(any(), any());
        }
    }

    @SneakyThrows
    public void testMergeOneField_whenBelowThreshold_thenSkipsBuild() {
        FaissBBQ1040KnnVectorsWriter w = new FaissBBQ1040KnnVectorsWriter(segmentWriteState, flatVectorsWriter, 100);
        final FieldInfo fi = fieldInfo(
            0,
            VectorEncoding.FLOAT32,
            Map.of(KNNConstants.VECTOR_DATA_TYPE_FIELD, "float", KNNConstants.KNN_ENGINE, "faiss")
        );
        final KNNVectorValues<float[]> kv = KNNVectorValuesFactory.getVectorValues(
            VectorDataType.FLOAT,
            new TestVectorValues.PreDefinedFloatVectorValues(List.of(new float[] { 1, 2, 3 }))
        );
        final Supplier<KNNVectorValues<?>> supplier = () -> kv;

        try (MockedStatic<KNNVectorValuesFactory> kvf = mockStatic(KNNVectorValuesFactory.class)) {
            kvf.when(() -> KNNVectorValuesFactory.getKNNVectorValuesSupplierForMerge(any(), any(), any())).thenReturn(supplier);
            w.mergeOneField(fi, mock(MergeState.class));
            verify(flatVectorsWriter).mergeOneField(any(), any());
        }
    }

    @SneakyThrows
    public void testMergeOneField_whenAboveThreshold_thenBuildsIndex() {
        final FieldInfo fi = fieldInfo(
            0,
            VectorEncoding.FLOAT32,
            Map.of(KNNConstants.VECTOR_DATA_TYPE_FIELD, "float", KNNConstants.KNN_ENGINE, "faiss")
        );
        final KNNVectorValues<float[]> kv = KNNVectorValuesFactory.getVectorValues(
            VectorDataType.FLOAT,
            new TestVectorValues.PreDefinedFloatVectorValues(List.of(new float[] { 1, 2, 3 }, new float[] { 4, 5, 6 }))
        );
        final Supplier<KNNVectorValues<?>> supplier = () -> kv;

        try (
            MockedStatic<KNNVectorValuesFactory> kvf = mockStatic(KNNVectorValuesFactory.class);
            MockedStatic<NativeIndexWriter> niw = mockStatic(NativeIndexWriter.class)
        ) {
            kvf.when(() -> KNNVectorValuesFactory.getKNNVectorValuesSupplierForMerge(any(), any(), any())).thenReturn(supplier);
            niw.when(() -> NativeIndexWriter.getWriter(any(FieldInfo.class), any(SegmentWriteState.class))).thenReturn(nativeIndexWriter);
            doAnswer(a -> {
                Thread.sleep(2);
                return null;
            }).when(nativeIndexWriter).mergeIndex(any(), anyInt());

            objectUnderTest.mergeOneField(fi, mock(MergeState.class));

            verify(flatVectorsWriter).mergeOneField(any(), any());
            verify(nativeIndexWriter).mergeIndex(supplier, 2);
            assertNotEquals(0L, (long) KNNGraphValue.MERGE_TOTAL_TIME_IN_MILLIS.getValue());
        }
    }

    // --- finish / close / ramBytesUsed ---

    @SneakyThrows
    public void testFinish_thenDelegatesToFlatWriter() {
        objectUnderTest.finish();
        verify(flatVectorsWriter).finish();
    }

    @SneakyThrows
    public void testFinish_calledTwice_thenThrowsException() {
        objectUnderTest.finish();
        expectThrows(IllegalStateException.class, () -> objectUnderTest.finish());
    }

    @SneakyThrows
    public void testClose_thenClosesFlatWriter() {
        objectUnderTest.close();
        verify(flatVectorsWriter).close();
    }

    public void testRamBytesUsed_whenNoField_thenReturnsShallowSize() {
        when(flatVectorsWriter.ramBytesUsed()).thenReturn(100L);
        assertTrue(objectUnderTest.ramBytesUsed() > 100L);
    }

    // --- helpers ---

    private FieldInfo fieldInfo(int fieldNumber, VectorEncoding vectorEncoding, Map<String, String> attributes) {
        FieldInfo fi = mock(FieldInfo.class);
        when(fi.getFieldNumber()).thenReturn(fieldNumber);
        when(fi.getVectorEncoding()).thenReturn(vectorEncoding);
        when(fi.attributes()).thenReturn(attributes);
        when(fi.getName()).thenReturn("field" + fieldNumber);
        attributes.forEach((key, value) -> when(fi.getAttribute(key)).thenReturn(value));
        return fi;
    }

    private <T> NativeEngineFieldVectorsWriter nativeEngineFieldVectorsWriter(FieldInfo fieldInfo, Map<Integer, T> vectors) {
        NativeEngineFieldVectorsWriter w = mock(NativeEngineFieldVectorsWriter.class);
        FlatFieldVectorsWriter ffw = mock(FlatFieldVectorsWriter.class);
        DocsWithFieldSet dwfs = new DocsWithFieldSet();
        vectors.keySet().stream().sorted().forEach(dwfs::add);
        when(w.getFieldInfo()).thenReturn(fieldInfo);
        when(w.getVectors()).thenReturn(vectors);
        when(w.getFlatFieldVectorsWriter()).thenReturn(ffw);
        when(ffw.getDocsWithFieldSet()).thenReturn(dwfs);
        return w;
    }
}
