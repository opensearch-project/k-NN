/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
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
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.Version;
import org.mockito.Mockito;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.util.KNNVectorAsCollectionOfHalfFloatsSerializer;
import org.opensearch.knn.memoryoptsearch.faiss.MMapFloatVectorValues;
import org.opensearch.knn.memoryoptsearch.faiss.WrappedFloatVectorValues;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class MergeOptimizedHalfFloatVectorTests extends KNNTestCase {

    private static final String FIELD_NAME = "fp16_vector";
    private static final int DIMENSION = 8;

    @SneakyThrows
    public void testMerge_readsRawBytesAndNeverDecodes_whenSourceIsHalfFloat() {
        try (Directory dir = new ByteBuffersDirectory()) {
            float[][] vectors = { randomVector(), randomVector(), randomVector() };
            SegmentReadState readState = writeHalfFloatSegment(dir, "_0", vectors);

            try (FlatVectorsReader reader = new KNN1040HalfFloatFlatVectorsReader(readState, mockScorer())) {
                KNN1040HalfFloatFlatVectorsValues spiedValues = Mockito.spy(
                    (KNN1040HalfFloatFlatVectorsValues) reader.getFloatVectorValues(FIELD_NAME)
                );

                List<byte[]> merged = collectMergedBytes(mergeStateOver(dir, identityDocMaps(1), spiedValues));

                assertVectorsEqual(vectors, merged);
                Mockito.verify(spiedValues, Mockito.never()).vectorValue(Mockito.anyInt());
                Mockito.verify(spiedValues, Mockito.times(vectors.length))
                    .readRawVectorBytes(Mockito.anyInt(), Mockito.any(), Mockito.anyInt());
            }
        }
    }

    @SneakyThrows
    public void testMerge_fallsBackToEncoding_forNonHalfFloatSource() {
        try (Directory dir = new ByteBuffersDirectory()) {
            float[][] halfFloatVectors = { randomVector(), randomVector() };
            float[][] plainVectors = { randomVector() };
            SegmentReadState readState = writeHalfFloatSegment(dir, "_0", halfFloatVectors);

            try (FlatVectorsReader reader = new KNN1040HalfFloatFlatVectorsReader(readState, mockScorer())) {
                FloatVectorValues halfFloatValues = reader.getFloatVectorValues(FIELD_NAME);
                FloatVectorValues plainValues = plainFloatVectorValues(plainVectors);

                // The plain sub's docs land after the FP16 sub's in the merged segment.
                MergeState.DocMap[] docMaps = { docID -> docID, docID -> halfFloatVectors.length + docID };
                List<byte[]> merged = collectMergedBytes(mergeStateOver(dir, docMaps, halfFloatValues, plainValues));

                assertVectorsEqual(new float[][] { halfFloatVectors[0], halfFloatVectors[1], plainVectors[0] }, merged);
            }
        }
    }

    @SneakyThrows
    public void testMerge_readsRawBytes_throughMMapWrapper() {
        try (Directory dir = new MMapDirectory(createTempDir())) {
            float[][] vectors = { randomVector(), randomVector() };
            SegmentReadState readState = writeHalfFloatSegment(dir, "_0", vectors);

            try (FlatVectorsReader reader = new KNN1040HalfFloatFlatVectorsReader(readState, mockScorer())) {
                FloatVectorValues fromReader = reader.getFloatVectorValues(FIELD_NAME);
                KNN1040HalfFloatFlatVectorsValues base = fromReader instanceof MMapFloatVectorValues mmap
                    ? (KNN1040HalfFloatFlatVectorsValues) mmap.getDelegate()
                    : (KNN1040HalfFloatFlatVectorsValues) fromReader;
                KNN1040HalfFloatFlatVectorsValues spiedValues = Mockito.spy(base);
                MMapFloatVectorValues wrapped = new MMapFloatVectorValues(spiedValues, new long[] { 1L, 1L });

                List<byte[]> merged = collectMergedBytes(mergeStateOver(dir, identityDocMaps(1), wrapped));

                assertVectorsEqual(vectors, merged);
                Mockito.verify(spiedValues, Mockito.never()).vectorValue(Mockito.anyInt());
            }
        }
    }

    @SneakyThrows
    public void testMerge_ordersVectorsByMappedDocId_underIndexSort() {
        try (Directory dir = new ByteBuffersDirectory()) {
            float[][] firstVectors = { randomVector() };
            float[][] secondVectors = { randomVector() };
            SegmentReadState firstState = writeHalfFloatSegment(dir, "_0", firstVectors);
            SegmentReadState secondState = writeHalfFloatSegment(dir, "_1", secondVectors);

            try (
                FlatVectorsReader firstReader = new KNN1040HalfFloatFlatVectorsReader(firstState, mockScorer());
                FlatVectorsReader secondReader = new KNN1040HalfFloatFlatVectorsReader(secondState, mockScorer())
            ) {
                // The second segment's doc sorts ahead of the first's, as an index sort could arrange.
                MergeState.DocMap[] docMaps = { docID -> docID + 1, docID -> docID };
                MergeState mergeState = mergeStateOver(
                    dir,
                    docMaps,
                    true,
                    firstReader.getFloatVectorValues(FIELD_NAME),
                    secondReader.getFloatVectorValues(FIELD_NAME)
                );

                assertVectorsEqual(new float[][] { secondVectors[0], firstVectors[0] }, collectMergedBytes(mergeState));
            }
        }
    }

    @SneakyThrows
    public void testCurrentVectorBytes_throwsBeforeIteration() {
        try (Directory dir = new ByteBuffersDirectory()) {
            SegmentReadState readState = writeHalfFloatSegment(dir, "_0", new float[][] { randomVector() });

            try (FlatVectorsReader reader = new KNN1040HalfFloatFlatVectorsReader(readState, mockScorer())) {
                MergeState mergeState = mergeStateOver(dir, identityDocMaps(1), reader.getFloatVectorValues(FIELD_NAME));
                MergeOptimizedHalfFloatVector values = MergeOptimizedHalfFloatVector.create(createFieldInfo(), mergeState);

                IllegalStateException e = expectThrows(
                    IllegalStateException.class,
                    () -> values.currentVectorBytes(new byte[DIMENSION * Short.BYTES])
                );
                assertTrue(e.getMessage().contains("nextDoc"));
            }
        }
    }

    @SneakyThrows
    public void testCreate_skipsSegmentsWithNothingToMerge() {
        try (Directory dir = new ByteBuffersDirectory()) {
            float[][] vectors = { randomVector(), randomVector() };
            SegmentReadState readState = writeHalfFloatSegment(dir, "_0", vectors);

            try (FlatVectorsReader reader = new KNN1040HalfFloatFlatVectorsReader(readState, mockScorer())) {
                KnnVectorsReader present = Mockito.mock(KnnVectorsReader.class);
                Mockito.when(present.getFloatVectorValues(FIELD_NAME)).thenReturn(reader.getFloatVectorValues(FIELD_NAME));
                KnnVectorsReader withoutValues = Mockito.mock(KnnVectorsReader.class);
                Mockito.when(withoutValues.getFloatVectorValues(FIELD_NAME)).thenReturn(null);

                FieldInfos withField = new FieldInfos(new FieldInfo[] { createFieldInfo() });
                FieldInfos withoutField = new FieldInfos(new FieldInfo[0]);

                // Only the first segment contributes: the second never had the field, the third has no
                // vectors reader at all, and the fourth has a reader but no values for this field.
                MergeState mergeState = mergeStateOver(
                    dir,
                    identityDocMaps(4),
                    new FieldInfos[] { withField, withoutField, withField, withField },
                    new KnnVectorsReader[] { present, present, null, withoutValues },
                    new int[] { vectors.length, 0, 0, 0 },
                    false
                );

                assertVectorsEqual(vectors, collectMergedBytes(mergeState));
            }
        }
    }

    @SneakyThrows
    public void testIterator_reportsPositionAndCost_andRejectsAdvance() {
        try (Directory dir = new ByteBuffersDirectory()) {
            float[][] vectors = { randomVector(), randomVector() };
            SegmentReadState readState = writeHalfFloatSegment(dir, "_0", vectors);

            try (FlatVectorsReader reader = new KNN1040HalfFloatFlatVectorsReader(readState, mockScorer())) {
                MergeOptimizedHalfFloatVector values = mergedViewOver(dir, reader.getFloatVectorValues(FIELD_NAME));
                KnnVectorValues.DocIndexIterator iterator = values.iterator();

                assertEquals(-1, iterator.docID());
                assertEquals(-1, iterator.index());
                assertEquals(vectors.length, iterator.cost());

                assertEquals(0, iterator.nextDoc());
                assertEquals(0, iterator.docID());
                assertEquals(0, iterator.index());

                assertEquals(1, iterator.nextDoc());
                assertEquals(1, iterator.docID());
                assertEquals(1, iterator.index());

                assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());
                assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.docID());
                assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.index());

                expectThrows(UnsupportedOperationException.class, () -> iterator.advance(0));
            }
        }
    }

    @SneakyThrows
    public void testSizeAndDimension_spanEverySub() {
        try (Directory dir = new ByteBuffersDirectory()) {
            SegmentReadState readState = writeHalfFloatSegment(dir, "_0", new float[][] { randomVector(), randomVector() });

            try (FlatVectorsReader reader = new KNN1040HalfFloatFlatVectorsReader(readState, mockScorer())) {
                MergeOptimizedHalfFloatVector values = mergedViewOver(
                    dir,
                    reader.getFloatVectorValues(FIELD_NAME),
                    plainFloatVectorValues(new float[][] { randomVector() })
                );

                assertEquals(3, values.size());
                assertEquals(DIMENSION, values.dimension());
            }
        }
    }

    @SneakyThrows
    public void testRandomAccessOperationsAreUnsupported() {
        try (Directory dir = new ByteBuffersDirectory()) {
            SegmentReadState readState = writeHalfFloatSegment(dir, "_0", new float[][] { randomVector() });

            try (FlatVectorsReader reader = new KNN1040HalfFloatFlatVectorsReader(readState, mockScorer())) {
                MergeOptimizedHalfFloatVector values = mergedViewOver(dir, reader.getFloatVectorValues(FIELD_NAME));

                // This view exists only to stream a merge forward; nothing on the merge path needs these.
                expectThrows(UnsupportedOperationException.class, () -> values.ordToDoc(0));
                expectThrows(UnsupportedOperationException.class, () -> values.scorer(randomVector()));
                expectThrows(UnsupportedOperationException.class, values::copy);
            }
        }
    }

    @SneakyThrows
    public void testVectorValue_decodesOnlyTheOrdTheIteratorIsOn() {
        try (Directory dir = new ByteBuffersDirectory()) {
            float[][] vectors = { randomVector(), randomVector() };
            SegmentReadState readState = writeHalfFloatSegment(dir, "_0", vectors);

            try (FlatVectorsReader reader = new KNN1040HalfFloatFlatVectorsReader(readState, mockScorer())) {
                MergeOptimizedHalfFloatVector values = mergedViewOver(dir, reader.getFloatVectorValues(FIELD_NAME));
                KnnVectorValues.DocIndexIterator iterator = values.iterator();
                iterator.nextDoc();

                assertArrayEquals(roundTripped(vectors[0]), values.vectorValue(0), 0.0f);

                IllegalStateException e = expectThrows(IllegalStateException.class, () -> values.vectorValue(1));
                assertTrue(e.getMessage().contains("forward iteration"));
            }
        }
    }

    @SneakyThrows
    public void testMerge_fallsBackToEncoding_forUnrecognisedWrappers() {
        try (Directory dir = new ByteBuffersDirectory()) {
            float[][] first = { randomVector() };
            float[][] second = { randomVector() };

            FloatVectorValues plain = plainFloatVectorValues(first);
            FloatVectorValues overPlain = new UnrecognisedWrapper(plain, plain);
            FloatVectorValues overNothing = new UnrecognisedWrapper(null, plainFloatVectorValues(second));

            MergeState.DocMap[] docMaps = { docID -> docID, docID -> first.length + docID };

            assertVectorsEqual(
                new float[][] { first[0], second[0] },
                collectMergedBytes(mergeStateOver(dir, docMaps, overPlain, overNothing))
            );
        }
    }

    @SneakyThrows
    public void testMerge_fallsBackToEncoding_whenSourceDimensionDisagrees() {
        try (Directory dir = new ByteBuffersDirectory()) {
            float[][] vectors = { randomVector() };
            SegmentReadState readState = writeHalfFloatSegment(dir, "_0", vectors);

            try (FlatVectorsReader reader = new KNN1040HalfFloatFlatVectorsReader(readState, mockScorer())) {
                KNN1040HalfFloatFlatVectorsValues spiedValues = Mockito.spy(
                    (KNN1040HalfFloatFlatVectorsValues) reader.getFloatVectorValues(FIELD_NAME)
                );
                Mockito.doReturn(DIMENSION + 1).when(spiedValues).dimension();

                assertVectorsEqual(vectors, collectMergedBytes(mergeStateOver(dir, identityDocMaps(1), spiedValues)));
                Mockito.verify(spiedValues).vectorValue(Mockito.anyInt());
            }
        }
    }

    /** Builds the merged view the writer would hand its vectors to. */
    private MergeOptimizedHalfFloatVector mergedViewOver(Directory dir, FloatVectorValues... values) throws IOException {
        return MergeOptimizedHalfFloatVector.create(createFieldInfo(), mergeStateOver(dir, identityDocMaps(values.length), values));
    }

    /** The vector as it survives a trip through FP16, which is what a decode off disk returns. */
    private float[] roundTripped(float[] vector) {
        byte[] encoded = new byte[DIMENSION * Short.BYTES];
        KNNVectorAsCollectionOfHalfFloatsSerializer.INSTANCE.floatToByteArray(vector, encoded, DIMENSION);
        float[] decoded = new float[DIMENSION];
        KNNVectorAsCollectionOfHalfFloatsSerializer.INSTANCE.byteToFloatArray(encoded, decoded, DIMENSION, 0);
        return decoded;
    }

    /** A wrapper the unwrap doesn't know how to see through; {@code inner} is the layer it exposes. */
    private static class UnrecognisedWrapper extends WrappedFloatVectorValues {
        private final FloatVectorValues source;

        UnrecognisedWrapper(FloatVectorValues inner, FloatVectorValues source) {
            super(inner);
            this.source = source;
        }

        @Override
        public int dimension() {
            return source.dimension();
        }

        @Override
        public int size() {
            return source.size();
        }

        @Override
        public float[] vectorValue(int ord) throws IOException {
            return source.vectorValue(ord);
        }

        @Override
        public FloatVectorValues copy() {
            return this;
        }

        @Override
        public DocIndexIterator iterator() {
            return source.iterator();
        }
    }

    /** Drives the merged view the way the writer does, collecting each vector's FP16 bytes. */
    private List<byte[]> collectMergedBytes(MergeState mergeState) throws IOException {
        MergeOptimizedHalfFloatVector values = MergeOptimizedHalfFloatVector.create(createFieldInfo(), mergeState);
        List<byte[]> collected = new ArrayList<>();
        KnnVectorValues.DocIndexIterator iterator = values.iterator();
        for (int doc = iterator.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = iterator.nextDoc()) {
            byte[] buffer = new byte[DIMENSION * Short.BYTES];
            values.currentVectorBytes(buffer);
            collected.add(buffer);
        }
        return collected;
    }

    private void assertVectorsEqual(float[][] expected, List<byte[]> actual) {
        assertEquals("vector count", expected.length, actual.size());
        byte[] expectedBytes = new byte[DIMENSION * Short.BYTES];
        for (int i = 0; i < expected.length; i++) {
            KNNVectorAsCollectionOfHalfFloatsSerializer.INSTANCE.floatToByteArray(expected[i], expectedBytes, DIMENSION);
            assertArrayEquals("vector " + i, expectedBytes, actual.get(i));
        }
    }

    private MergeState mergeStateOver(Directory dir, MergeState.DocMap[] docMaps, FloatVectorValues... values) throws IOException {
        return mergeStateOver(dir, docMaps, false, values);
    }

    /**
     * @param needsIndexSort selects {@link org.apache.lucene.index.DocIDMerger}'s sorted merger, which
     *                       orders vectors by mapped doc id; without it subs are simply concatenated.
     */
    private MergeState mergeStateOver(Directory dir, MergeState.DocMap[] docMaps, boolean needsIndexSort, FloatVectorValues... values)
        throws IOException {
        FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { createFieldInfo() });
        KnnVectorsReader[] readers = new KnnVectorsReader[values.length];
        FieldInfos[] sourceFieldInfos = new FieldInfos[values.length];
        int[] maxDocs = new int[values.length];
        for (int i = 0; i < values.length; i++) {
            KnnVectorsReader reader = Mockito.mock(KnnVectorsReader.class);
            Mockito.when(reader.getFloatVectorValues(FIELD_NAME)).thenReturn(values[i]);
            readers[i] = reader;
            sourceFieldInfos[i] = fieldInfos;
            maxDocs[i] = values[i].size();
        }

        return mergeStateOver(dir, docMaps, sourceFieldInfos, readers, maxDocs, needsIndexSort);
    }

    /** Spells out each segment's field infos and reader, so segments the merge must skip can be posed. */
    private MergeState mergeStateOver(
        Directory dir,
        MergeState.DocMap[] docMaps,
        FieldInfos[] sourceFieldInfos,
        KnnVectorsReader[] readers,
        int[] maxDocs,
        boolean needsIndexSort
    ) {
        int totalDocs = 0;
        for (int maxDoc : maxDocs) {
            totalDocs += maxDoc;
        }

        return new MergeState(
            docMaps,
            createSegmentInfo(dir, "_merged", totalDocs),
            new FieldInfos(new FieldInfo[] { createFieldInfo() }),
            null,
            null,
            null,
            null,
            sourceFieldInfos,
            new Bits[readers.length],
            null,
            null,
            readers,
            maxDocs,
            InfoStream.NO_OUTPUT,
            null,
            needsIndexSort,
            null
        );
    }

    private MergeState.DocMap[] identityDocMaps(int count) {
        MergeState.DocMap[] docMaps = new MergeState.DocMap[count];
        for (int i = 0; i < count; i++) {
            docMaps[i] = docID -> docID;
        }
        return docMaps;
    }

    @SuppressWarnings("unchecked")
    private SegmentReadState writeHalfFloatSegment(Directory dir, String segmentName, float[][] vectors) throws IOException {
        FieldInfo fieldInfo = createFieldInfo();
        FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { fieldInfo });
        SegmentInfo segmentInfo = createSegmentInfo(dir, segmentName, vectors.length);
        SegmentWriteState writeState = new SegmentWriteState(InfoStream.NO_OUTPUT, dir, segmentInfo, fieldInfos, null, IOContext.DEFAULT);

        try (FlatVectorsWriter writer = new KNN1040HalfFloatFlatVectorsWriter(writeState, mockScorer())) {
            FlatFieldVectorsWriter<float[]> fieldWriter = (FlatFieldVectorsWriter<float[]>) writer.addField(fieldInfo);
            for (int docID = 0; docID < vectors.length; docID++) {
                fieldWriter.addValue(docID, vectors[docID]);
            }
            writer.flush(vectors.length, null);
            writer.finish();
        }

        return new SegmentReadState(dir, segmentInfo, fieldInfos, IOContext.DEFAULT);
    }

    private FloatVectorValues plainFloatVectorValues(float[][] vectors) {
        return new FloatVectorValues() {
            @Override
            public int dimension() {
                return DIMENSION;
            }

            @Override
            public int size() {
                return vectors.length;
            }

            @Override
            public float[] vectorValue(int ord) {
                return vectors[ord];
            }

            @Override
            public FloatVectorValues copy() {
                return this;
            }

            @Override
            public DocIndexIterator iterator() {
                return createDenseIterator();
            }
        };
    }

    private SegmentInfo createSegmentInfo(Directory dir, String name, int maxDoc) {
        return new SegmentInfo(
            dir,
            Version.LATEST,
            Version.LATEST,
            name,
            maxDoc,
            false,
            false,
            null,
            Collections.emptyMap(),
            StringHelper.randomId(),
            new HashMap<>(),
            null
        );
    }

    private FieldInfo createFieldInfo() {
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
            Map.of(),
            0,
            0,
            0,
            DIMENSION,
            VectorEncoding.FLOAT32,
            VectorSimilarityFunction.EUCLIDEAN,
            false,
            false
        );
    }

    private FlatVectorsScorer mockScorer() {
        return Mockito.mock(FlatVectorsScorer.class);
    }

    private float[] randomVector() {
        float[] vector = new float[DIMENSION];
        for (int i = 0; i < DIMENSION; i++) {
            vector[i] = randomFloat();
        }
        return vector;
    }
}
