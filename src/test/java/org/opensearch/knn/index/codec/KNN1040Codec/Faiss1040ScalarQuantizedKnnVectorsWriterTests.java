/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.util.IOFunction;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.Version;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategyFactory;

import java.io.IOException;
import java.util.Collections;
import java.util.Map;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

public class Faiss1040ScalarQuantizedKnnVectorsWriterTests extends KNNTestCase {

    private static final int DIMENSION = 128;
    private static final String FIELD_NAME = "test_field";
    private static final String PARAMETERS_JSON = "{"
        + "\"index_description\":\"BHNSW16,Flat\","
        + "\"spaceType\":\"innerproduct\","
        + "\"name\":\"hnsw\","
        + "\"data_type\":\"float\","
        + "\"parameters\":{"
        + "\"ef_search\":256,"
        + "\"ef_construction\":256,"
        + "\"m\":16,"
        + "\"encoder\":{\"name\":\"sq\",\"bits\":1}"
        + "}"
        + "}";

    @Mock
    private FlatVectorsWriter flatVectorsWriter;
    @Mock
    private SegmentWriteState segmentWriteState;
    @Mock
    private SegmentInfo segmentInfo;
    @Mock
    private IOFunction<SegmentReadState, FlatVectorsReader> quantizedFlatVectorsReaderSupplier;

    private FlatFieldVectorsWriter<?> mockedFlatFieldVectorsWriter;
    private Faiss1040ScalarQuantizedKnnVectorsWriter objectUnderTest;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        MockitoAnnotations.openMocks(this);
        objectUnderTest = new Faiss1040ScalarQuantizedKnnVectorsWriter(
            segmentWriteState,
            flatVectorsWriter,
            quantizedFlatVectorsReaderSupplier,
            new NativeIndexBuildStrategyFactory()
        );
        mockedFlatFieldVectorsWriter = mock(FlatFieldVectorsWriter.class);
        Mockito.doNothing().when(mockedFlatFieldVectorsWriter).addValue(Mockito.anyInt(), Mockito.any());
        when(flatVectorsWriter.addField(any())).thenReturn((FlatFieldVectorsWriter) mockedFlatFieldVectorsWriter);

        try {
            java.lang.reflect.Field infoField = SegmentWriteState.class.getDeclaredField("segmentInfo");
            infoField.setAccessible(true);
            infoField.set(segmentWriteState, segmentInfo);
        } catch (Exception ignored) {}
        when(segmentInfo.getVersion()).thenReturn(Version.LATEST);
    }

    // ===================== addField =====================

    /**
     * Test: addField delegates to the underlying flat vectors writer.
     *
     * Expected: flatVectorsWriter.addField(fi) is called exactly once.
     */
    @SneakyThrows
    public void testAddField_thenDelegatesToFlatWriter() {
        final FieldInfo fi = mockFieldInfo(0);
        objectUnderTest.addField(fi);
        verify(flatVectorsWriter).addField(fi);
    }

    /**
     * Test: addField called twice throws IllegalStateException with both field names.
     *
     * Expected: IllegalStateException containing both field names.
     */
    @SneakyThrows
    public void testAddField_calledTwice_thenThrowsIllegalState() {
        final FieldInfo fi1 = mockFieldInfo(0);
        final FieldInfo fi2 = mockFieldInfo(1);
        objectUnderTest.addField(fi1);
        IllegalStateException ex = expectThrows(IllegalStateException.class, () -> objectUnderTest.addField(fi2));
        assertTrue(ex.getMessage().contains(fi1.getName()));
        assertTrue(ex.getMessage().contains(fi2.getName()));
    }

    // ===================== flush (mocked — tests that don't reach openFlatVectorsReader) =====================

    /**
     * Test: flush with no field added delegates flush/finish/close to flat writer, skips native build.
     */
    @SneakyThrows
    public void testFlush_whenNoFieldAdded_thenOnlyDelegatesFlat() {
        objectUnderTest.flush(5, null);
        verify(flatVectorsWriter).flush(5, null);
        verify(flatVectorsWriter).finish();
        verify(flatVectorsWriter).close();
    }

    /**
     * Test: flush passes a non-null sortMap through to the flat writer.
     */
    @SneakyThrows
    public void testFlush_withSortMap_passesThroughToFlatWriter() {
        org.apache.lucene.index.Sorter.DocMap sortMap = mock(org.apache.lucene.index.Sorter.DocMap.class);
        objectUnderTest.flush(10, sortMap);
        verify(flatVectorsWriter).flush(10, sortMap);
    }

    /**
     * Test: IOException from flatVectorsWriter.flush propagates before native build starts.
     */
    @SneakyThrows
    public void testFlush_whenFlatWriterThrows_thenPropagatesBeforeNativeBuild() {
        final FieldInfo fi = mockFieldInfo(0);
        objectUnderTest.addField(fi);
        doThrow(new IOException("flat flush failed")).when(flatVectorsWriter).flush(anyInt(), any());
        expectThrows(IOException.class, () -> objectUnderTest.flush(5, null));
    }

    // ===================== mergeOneField (mocked — tests that don't reach openFlatVectorsReader) =====================

    /**
     * Test: IOException from flatVectorsWriter.mergeOneField propagates immediately.
     */
    @SneakyThrows
    public void testMergeOneField_whenFlatMergeThrows_thenPropagates() {
        final FieldInfo fi = mockFieldInfo(0);
        doThrow(new IOException("flat merge failed")).when(flatVectorsWriter).mergeOneField(any(), any());
        expectThrows(IOException.class, () -> objectUnderTest.mergeOneField(fi, mock(MergeState.class)));
    }

    // ===================== finish =====================

    /**
     * Test: finish sets the finished flag but does not call flatVectorsWriter.finish()
     * (that's already done in flush/mergeOneField).
     */
    @SneakyThrows
    public void testFinish_thenSetsFinishedFlag() {
        objectUnderTest.finish();
        verify(flatVectorsWriter, Mockito.never()).finish();
    }

    /**
     * Test: finish called twice throws IllegalStateException.
     */
    @SneakyThrows
    public void testFinish_calledTwice_thenThrowsIllegalState() {
        objectUnderTest.finish();
        expectThrows(IllegalStateException.class, () -> objectUnderTest.finish());
    }

    /**
     * Test: finish works even if no field was ever added.
     */
    @SneakyThrows
    public void testFinish_withoutAddField_thenSucceeds() {
        objectUnderTest.finish();
        verify(flatVectorsWriter, Mockito.never()).finish();
    }

    // ===================== close =====================

    /**
     * Test: close delegates to IOUtils.close(flatVectorsWriter).
     */
    @SneakyThrows
    public void testClose_thenClosesFlatWriter() {
        objectUnderTest.close();
        verify(flatVectorsWriter).close();
    }

    /**
     * Test: IOException from flatVectorsWriter.close() propagates.
     */
    @SneakyThrows
    public void testClose_whenFlatWriterThrows_thenPropagates() {
        doThrow(new IOException("close failed")).when(flatVectorsWriter).close();
        expectThrows(IOException.class, () -> objectUnderTest.close());
    }

    // ===================== ramBytesUsed =====================

    /**
     * Test: ramBytesUsed without a field returns SHALLOW_SIZE + flat writer's usage.
     */
    public void testRamBytesUsed_whenNoField_thenReturnsShallowPlusFlatWriter() {
        when(flatVectorsWriter.ramBytesUsed()).thenReturn(100L);
        assertTrue(objectUnderTest.ramBytesUsed() > 100L);
    }

    /**
     * Test: ramBytesUsed with a field includes the field writer's memory usage.
     */
    @SneakyThrows
    public void testRamBytesUsed_whenFieldAdded_thenIncludesFieldWriter() {
        final FieldInfo fi = mockFieldInfo(0);
        when(flatVectorsWriter.ramBytesUsed()).thenReturn(100L);
        when(mockedFlatFieldVectorsWriter.ramBytesUsed()).thenReturn(50L);
        objectUnderTest.addField(fi);
        assertTrue(objectUnderTest.ramBytesUsed() > 150L);
    }

    // ===================== integration: real flush =====================

    /**
     * Test: ingest 345 real 128-dim vectors, flush, and verify flat vectors are readable.
     * Uses real Lucene flat file I/O and real JNI native HNSW graph build (no mocks).
     */
    @SneakyThrows
    public void testFlush_whenRealVectorsIngested_thenFlatVectorsWrittenCorrectly() {
        final int numVectors = 345;
        final float[][] vectors = generateRandomVectors(numVectors, DIMENSION);

        try (org.apache.lucene.store.Directory directory = newDirectory()) {
            final SegmentWriteState writeState = createWriteState(directory, "_0", numVectors);
            final FieldInfo fi = createRealFieldInfo();
            final Faiss1040ScalarQuantizedKnnVectorsFormat format = new Faiss1040ScalarQuantizedKnnVectorsFormat();

            try (KnnVectorsWriter knnWriter = format.fieldsWriter(writeState)) {
                KnnFieldVectorsWriter<float[]> fw = (KnnFieldVectorsWriter<float[]>) knnWriter.addField(fi);
                for (int i = 0; i < numVectors; i++) {
                    fw.addValue(i, vectors[i]);
                }
                knnWriter.flush(numVectors, null);
                knnWriter.finish();
            }

            verifyVectorsReadable(format, directory, writeState, fi, vectors, numVectors);
        }
    }

    // ===================== integration: real merge =====================

    /**
     * Test: merge 3 segments of 150 vectors each, verify all 450 vectors are readable.
     * Uses real Lucene flat file I/O and real JNI native HNSW graph build (no mocks).
     */
    @SneakyThrows
    public void testMergeOneField_when3SegmentsMerged_thenAllVectorsReadable() {
        final int numSegments = 3;
        final int vectorsPerSegment = 150;
        final int totalVectors = numSegments * vectorsPerSegment;

        final float[][][] segmentVectors = new float[numSegments][][];
        for (int s = 0; s < numSegments; s++) {
            segmentVectors[s] = generateRandomVectors(vectorsPerSegment, DIMENSION);
        }

        final FieldInfo fi = createRealFieldInfo();
        final FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { fi });
        final Faiss1040ScalarQuantizedKnnVectorsFormat format = new Faiss1040ScalarQuantizedKnnVectorsFormat();

        try (org.apache.lucene.store.Directory directory = newDirectory()) {
            // Step 1: Create 3 segments by flushing vectors
            final SegmentInfo[] segInfos = new SegmentInfo[numSegments];
            final KnnVectorsReader[] readers = new KnnVectorsReader[numSegments];

            for (int s = 0; s < numSegments; s++) {
                segInfos[s] = createSegmentInfo(directory, "_" + s, vectorsPerSegment);
                final SegmentWriteState ws = new SegmentWriteState(
                    InfoStream.NO_OUTPUT,
                    directory,
                    segInfos[s],
                    fieldInfos,
                    null,
                    IOContext.DEFAULT,
                    FIELD_NAME
                );

                try (KnnVectorsWriter knnWriter = format.fieldsWriter(ws)) {
                    KnnFieldVectorsWriter<float[]> fw = (KnnFieldVectorsWriter<float[]>) knnWriter.addField(fi);
                    for (int i = 0; i < vectorsPerSegment; i++) {
                        fw.addValue(i, segmentVectors[s][i]);
                    }
                    knnWriter.flush(vectorsPerSegment, null);
                    knnWriter.finish();
                }

                readers[s] = format.fieldsReader(new SegmentReadState(directory, segInfos[s], fieldInfos, IOContext.DEFAULT, FIELD_NAME));
            }

            // Step 2: Build MergeState
            final MergeState.DocMap[] docMaps = new MergeState.DocMap[numSegments];
            final int[] maxDocs = new int[numSegments];
            final FieldInfos[] perSegFieldInfos = new FieldInfos[numSegments];
            int docBase = 0;
            for (int s = 0; s < numSegments; s++) {
                final int base = docBase;
                docMaps[s] = docID -> base + docID;
                maxDocs[s] = vectorsPerSegment;
                perSegFieldInfos[s] = fieldInfos;
                docBase += vectorsPerSegment;
            }

            final SegmentInfo mergedSegInfo = createSegmentInfo(directory, "_merged", totalVectors);
            final MergeState mergeState = new MergeState(
                docMaps,
                mergedSegInfo,
                fieldInfos,
                null,
                null,
                null,
                null,
                perSegFieldInfos,
                new org.apache.lucene.util.Bits[numSegments],
                null,
                null,
                readers,
                maxDocs,
                InfoStream.NO_OUTPUT,
                Runnable::run,
                false
            );

            // Step 3: Merge
            final SegmentWriteState mergedWriteState = new SegmentWriteState(
                InfoStream.NO_OUTPUT,
                directory,
                mergedSegInfo,
                fieldInfos,
                null,
                IOContext.DEFAULT,
                FIELD_NAME
            );

            try (KnnVectorsWriter mergedWriter = format.fieldsWriter(mergedWriteState)) {
                mergedWriter.mergeOneField(fi, mergeState);
                mergedWriter.finish();
            }

            for (KnnVectorsReader reader : readers) {
                reader.close();
            }

            // Step 4: Verify all vectors
            final float[][] allVectors = new float[totalVectors][];
            for (int s = 0; s < numSegments; s++) {
                System.arraycopy(segmentVectors[s], 0, allVectors, s * vectorsPerSegment, vectorsPerSegment);
            }
            verifyVectorsReadable(format, directory, mergedWriteState, fi, allVectors, totalVectors);
        }
    }

    // ===================== helpers =====================

    /** Creates a mock FieldInfo for unit tests that don't need real Lucene I/O. */
    @SneakyThrows
    private FieldInfo mockFieldInfo(int fieldNumber) {
        FieldInfo fi = mock(FieldInfo.class);
        when(fi.getFieldNumber()).thenReturn(fieldNumber);
        when(fi.getVectorEncoding()).thenReturn(VectorEncoding.FLOAT32);
        when(fi.attributes()).thenReturn(Map.of(KNNConstants.VECTOR_DATA_TYPE_FIELD, "float"));
        when(fi.getName()).thenReturn("field" + fieldNumber);
        when(fi.getAttribute(KNNConstants.VECTOR_DATA_TYPE_FIELD)).thenReturn("float");

        java.lang.reflect.Field nameField = FieldInfo.class.getDeclaredField("name");
        nameField.setAccessible(true);
        nameField.set(fi, "field" + fieldNumber);

        return fi;
    }

    /** Creates a real FieldInfo with all required attributes for integration tests. */
    private FieldInfo createRealFieldInfo() {
        return new FieldInfo(
            FIELD_NAME,
            0,
            false,
            false,
            false,
            IndexOptions.NONE,
            DocValuesType.NONE,
            DocValuesSkipIndexType.NONE,
            -1,
            Map.of(
                KNNConstants.PARAMETERS,
                PARAMETERS_JSON,
                KNNConstants.VECTOR_DATA_TYPE_FIELD,
                "float",
                KNNConstants.KNN_ENGINE,
                "faiss",
                KNNConstants.SQ_CONFIG,
                "bits=1"
            ),
            0,
            0,
            0,
            DIMENSION,
            VectorEncoding.FLOAT32,
            VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT,
            false,
            false
        );
    }

    /** Creates a SegmentInfo for integration tests. */
    private SegmentInfo createSegmentInfo(org.apache.lucene.store.Directory directory, String segName, int maxDoc) {
        return new SegmentInfo(
            directory,
            Version.LATEST,
            Version.LATEST,
            segName,
            maxDoc,
            false,
            false,
            null,
            Collections.emptyMap(),
            StringHelper.randomId(),
            Collections.emptyMap(),
            null
        );
    }

    /** Creates a SegmentWriteState for integration tests. */
    private SegmentWriteState createWriteState(org.apache.lucene.store.Directory directory, String segName, int maxDoc) {
        final FieldInfo fi = createRealFieldInfo();
        final FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { fi });
        final SegmentInfo segInfo = createSegmentInfo(directory, segName, maxDoc);
        return new SegmentWriteState(InfoStream.NO_OUTPUT, directory, segInfo, fieldInfos, null, IOContext.DEFAULT, FIELD_NAME);
    }

    /** Generates random float vectors with a fixed seed for reproducibility. */
    private float[][] generateRandomVectors(int numVectors, int dimension) {
        final java.util.Random rng = new java.util.Random(42);
        final float[][] vectors = new float[numVectors][dimension];
        for (int i = 0; i < numVectors; i++) {
            for (int j = 0; j < dimension; j++) {
                vectors[i][j] = rng.nextFloat() * 2 - 1;
            }
        }
        return vectors;
    }

    /** Opens a reader on the written segment and verifies all vectors match the originals. */
    @SneakyThrows
    private void verifyVectorsReadable(
        Faiss1040ScalarQuantizedKnnVectorsFormat format,
        org.apache.lucene.store.Directory directory,
        SegmentWriteState writeState,
        FieldInfo fi,
        float[][] expectedVectors,
        int expectedCount
    ) {
        final SegmentReadState readState = new SegmentReadState(
            directory,
            writeState.segmentInfo,
            new FieldInfos(new FieldInfo[] { fi }),
            IOContext.DEFAULT,
            FIELD_NAME
        );

        try (KnnVectorsReader knnReader = format.fieldsReader(readState)) {
            FloatVectorValues fvv = knnReader.getFloatVectorValues(FIELD_NAME);
            assertNotNull(fvv);
            assertEquals(expectedCount, fvv.size());
            assertEquals(DIMENSION, fvv.dimension());

            KnnVectorValues.DocIndexIterator it = fvv.iterator();
            int count = 0;
            while (it.nextDoc() != KnnVectorValues.DocIndexIterator.NO_MORE_DOCS) {
                assertArrayEquals(expectedVectors[count], fvv.vectorValue(it.index()), 0.0f);
                count++;
            }
            assertEquals(expectedCount, count);
        }
    }
}
