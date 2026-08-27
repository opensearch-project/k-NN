/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FilterDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.Version;
import org.mockito.Mockito;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.util.KNNVectorAsCollectionOfHalfFloatsSerializer;

import java.io.IOException;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * Tests {@link KNN1040HalfFloatFlatVectorsWriter} in isolation, without going through
 * {@link KNN1040HalfFloatFlatVectorsReader}. Correctness of the encoded bytes is verified by decoding
 * them directly with {@link KNNVectorAsCollectionOfHalfFloatsSerializer} (a shared utility, not part of
 * the writer/reader pair), rather than by reading the segment back through our own reader.
 */
public class KNN1040HalfFloatFlatVectorsWriterTests extends KNNTestCase {

    private static final String FIELD_NAME = "fp16_vector";
    private static final int DIMENSION = 8;

    @SneakyThrows
    public void testAddField_returnsWorkingFieldWriter() {
        try (Directory dir = new ByteBuffersDirectory()) {
            try (FlatVectorsWriter writer = newWriter(dir, "_0")) {
                FieldInfo fieldInfo = createFieldInfo();
                FlatFieldVectorsWriter<?> fieldWriter = writer.addField(fieldInfo);
                assertNotNull(fieldWriter);
                assertTrue(fieldWriter.getVectors().isEmpty());
                assertEquals(0, fieldWriter.getDocsWithFieldSet().cardinality());
            }
        }
    }

    @SneakyThrows
    public void testAddField_rejectsNonFloat32Encoding() {
        try (Directory dir = new ByteBuffersDirectory()) {
            try (FlatVectorsWriter writer = newWriter(dir, "_0")) {
                FieldInfo byteEncodedField = createFieldInfo(VectorEncoding.BYTE, VectorSimilarityFunction.EUCLIDEAN);
                IllegalArgumentException e = expectThrows(IllegalArgumentException.class, () -> writer.addField(byteEncodedField));
                assertTrue(e.getMessage().contains("FLOAT32"));
            }
        }
    }

    @SneakyThrows
    public void testFieldWriter_duplicateDocId_throws() {
        try (Directory dir = new ByteBuffersDirectory()) {
            try (FlatVectorsWriter writer = newWriter(dir, "_0")) {
                @SuppressWarnings("unchecked")
                FlatFieldVectorsWriter<float[]> fieldWriter = (FlatFieldVectorsWriter<float[]>) writer.addField(createFieldInfo());
                fieldWriter.addValue(0, new float[DIMENSION]);
                IllegalArgumentException e = expectThrows(
                    IllegalArgumentException.class,
                    () -> fieldWriter.addValue(0, new float[DIMENSION])
                );
                assertTrue(e.getMessage().contains(FIELD_NAME));
            }
        }
    }

    @SneakyThrows
    public void testFieldWriter_addValueAfterFinish_throws() {
        try (Directory dir = new ByteBuffersDirectory()) {
            try (FlatVectorsWriter writer = newWriter(dir, "_0")) {
                @SuppressWarnings("unchecked")
                FlatFieldVectorsWriter<float[]> fieldWriter = (FlatFieldVectorsWriter<float[]>) writer.addField(createFieldInfo());
                fieldWriter.finish();
                assertTrue(fieldWriter.isFinished());
                expectThrows(IllegalStateException.class, () -> fieldWriter.addValue(0, new float[DIMENSION]));
            }
        }
    }

    @SneakyThrows
    public void testFinish_calledTwice_throws() {
        try (Directory dir = new ByteBuffersDirectory()) {
            try (FlatVectorsWriter writer = newWriter(dir, "_0")) {
                writer.finish();
                expectThrows(IllegalStateException.class, writer::finish);
            }
        }
    }

    @SneakyThrows
    public void testRamBytesUsed_increasesAsVectorsAreAdded() {
        try (Directory dir = new ByteBuffersDirectory()) {
            try (FlatVectorsWriter writer = newWriter(dir, "_0")) {
                long before = writer.ramBytesUsed();
                @SuppressWarnings("unchecked")
                FlatFieldVectorsWriter<float[]> fieldWriter = (FlatFieldVectorsWriter<float[]>) writer.addField(createFieldInfo());
                for (int i = 0; i < 5; i++) {
                    fieldWriter.addValue(i, generateVector());
                }
                long after = writer.ramBytesUsed();
                assertTrue("ramBytesUsed should grow as vectors are added", after > before);
            }
        }
    }

    @SneakyThrows
    public void testFlush_writesFilesWithExpectedNamesAndHeaders() {
        try (Directory dir = new ByteBuffersDirectory()) {
            String segmentSuffix = "";
            FieldInfo fieldInfo = createFieldInfo();
            try (FlatVectorsWriter writer = newWriter(dir, "_0")) {
                @SuppressWarnings("unchecked")
                FlatFieldVectorsWriter<float[]> fieldWriter = (FlatFieldVectorsWriter<float[]>) writer.addField(fieldInfo);
                fieldWriter.addValue(0, generateVector());
                writer.flush(1, null);
                writer.finish();
            }

            String metaFileName = IndexFileNames.segmentFileName("_0", segmentSuffix, KNN1040HalfFloatFlatVectorsFormat.META_EXTENSION);
            String vectorDataFileName = IndexFileNames.segmentFileName(
                "_0",
                segmentSuffix,
                KNN1040HalfFloatFlatVectorsFormat.VECTOR_DATA_EXTENSION
            );
            assertTrue(java.util.Arrays.asList(dir.listAll()).contains(metaFileName));
            assertTrue(java.util.Arrays.asList(dir.listAll()).contains(vectorDataFileName));

            try (IndexInput metaInput = dir.openInput(metaFileName, IOContext.DEFAULT)) {
                CodecUtil.checksumEntireFile(metaInput);
            }
            try (IndexInput vectorDataInput = dir.openInput(vectorDataFileName, IOContext.DEFAULT)) {
                CodecUtil.checksumEntireFile(vectorDataInput);
            }
        }
    }

    /**
     * Decodes the written FP16 bytes directly with {@link KNNVectorAsCollectionOfHalfFloatsSerializer},
     * independent of {@link KNN1040HalfFloatFlatVectorsReader}, to isolate encoding correctness in the
     * writer from any bugs the reader might share or mask.
     */
    @SneakyThrows
    public void testVectorDataBytes_decodeToExpectedFp16Values() {
        try (Directory dir = new ByteBuffersDirectory()) {
            float[][] vectors = { generateVector(), generateVector(), generateVector() };
            FieldInfo fieldInfo = createFieldInfo();
            SegmentInfo segmentInfo = createSegmentInfo(dir, "_0");

            try (FlatVectorsWriter writer = newWriter(dir, segmentInfo)) {
                @SuppressWarnings("unchecked")
                FlatFieldVectorsWriter<float[]> fieldWriter = (FlatFieldVectorsWriter<float[]>) writer.addField(fieldInfo);
                for (int i = 0; i < vectors.length; i++) {
                    fieldWriter.addValue(i, vectors[i]);
                }
                writer.flush(vectors.length, null);
                writer.finish();
            }

            String vectorDataFileName = IndexFileNames.segmentFileName("_0", "", KNN1040HalfFloatFlatVectorsFormat.VECTOR_DATA_EXTENSION);
            try (IndexInput input = dir.openInput(vectorDataFileName, IOContext.DEFAULT)) {
                CodecUtil.checkIndexHeader(
                    input,
                    KNN1040HalfFloatFlatVectorsFormat.VECTOR_DATA_CODEC_NAME,
                    KNN1040HalfFloatFlatVectorsFormat.VERSION_CURRENT,
                    KNN1040HalfFloatFlatVectorsFormat.VERSION_CURRENT,
                    segmentInfo.getId(),
                    ""
                );
                // The writer 64-byte-aligns the start of each field's vector data (including the
                // first), via IndexOutput#alignFilePointer - mirror that here before reading.
                final int vectorDataAlignment = 64;
                long aligned = ((input.getFilePointer() + vectorDataAlignment - 1) / vectorDataAlignment) * vectorDataAlignment;
                input.seek(aligned);

                int byteSize = DIMENSION * Short.BYTES;
                byte[] raw = new byte[byteSize];
                for (float[] expected : vectors) {
                    input.readBytes(raw, 0, byteSize);
                    float[] decoded = KNNVectorAsCollectionOfHalfFloatsSerializer.INSTANCE.byteToFloatArray(
                        new org.apache.lucene.util.BytesRef(raw)
                    );
                    for (int d = 0; d < DIMENSION; d++) {
                        float expectedFp16 = Float.float16ToFloat(Float.floatToFloat16(expected[d]));
                        assertEquals(expectedFp16, decoded[d], 0.0f);
                    }
                }
            }
        }
    }

    @SneakyThrows
    public void testFlush_withSortMap_writesVectorsInNewDocOrder() {
        try (Directory dir = new ByteBuffersDirectory()) {
            float[][] vectors = { generateVector(), generateVector(), generateVector() };
            FieldInfo fieldInfo = createFieldInfo();
            SegmentInfo segmentInfo = createSegmentInfo(dir, "_0");
            // Reverses doc order: old doc i becomes new doc (vectors.length - 1 - i).
            Sorter.DocMap sortMap = new Sorter.DocMap() {
                @Override
                public int oldToNew(int docID) {
                    return vectors.length - 1 - docID;
                }

                @Override
                public int newToOld(int docID) {
                    return vectors.length - 1 - docID;
                }

                @Override
                public int size() {
                    return vectors.length;
                }
            };

            try (FlatVectorsWriter writer = newWriter(dir, segmentInfo)) {
                @SuppressWarnings("unchecked")
                FlatFieldVectorsWriter<float[]> fieldWriter = (FlatFieldVectorsWriter<float[]>) writer.addField(fieldInfo);
                for (int i = 0; i < vectors.length; i++) {
                    fieldWriter.addValue(i, vectors[i]);
                }
                writer.flush(vectors.length, sortMap);
                writer.finish();
            }

            String vectorDataFileName = IndexFileNames.segmentFileName("_0", "", KNN1040HalfFloatFlatVectorsFormat.VECTOR_DATA_EXTENSION);
            try (IndexInput input = dir.openInput(vectorDataFileName, IOContext.DEFAULT)) {
                CodecUtil.checkIndexHeader(
                    input,
                    KNN1040HalfFloatFlatVectorsFormat.VECTOR_DATA_CODEC_NAME,
                    KNN1040HalfFloatFlatVectorsFormat.VERSION_CURRENT,
                    KNN1040HalfFloatFlatVectorsFormat.VERSION_CURRENT,
                    segmentInfo.getId(),
                    ""
                );
                final int vectorDataAlignment = 64;
                long aligned = ((input.getFilePointer() + vectorDataAlignment - 1) / vectorDataAlignment) * vectorDataAlignment;
                input.seek(aligned);

                int byteSize = DIMENSION * Short.BYTES;
                byte[] raw = new byte[byteSize];
                // New doc order is the reverse of vectors[]: expect vectors[2], vectors[1], vectors[0].
                for (int newDoc = 0; newDoc < vectors.length; newDoc++) {
                    input.readBytes(raw, 0, byteSize);
                    float[] decoded = KNNVectorAsCollectionOfHalfFloatsSerializer.INSTANCE.byteToFloatArray(
                        new org.apache.lucene.util.BytesRef(raw)
                    );
                    float[] expected = vectors[vectors.length - 1 - newDoc];
                    for (int d = 0; d < DIMENSION; d++) {
                        float expectedFp16 = Float.float16ToFloat(Float.floatToFloat16(expected[d]));
                        assertEquals("newDoc " + newDoc + " dim " + d, expectedFp16, decoded[d], 0.0f);
                    }
                }
            }
        }
    }

    // Forces the second createOutput() call (vectorData) to fail after the first (meta) already
    // succeeded, exercising the constructor's `if (!success) IOUtils.closeWhileHandlingException(this)`
    // cleanup path.
    @SneakyThrows
    public void testConstructor_directoryFailsCreatingVectorData_cleansUpAndPropagatesOriginalException() {
        try (Directory baseDir = new ByteBuffersDirectory()) {
            Directory failingDir = new FilterDirectory(baseDir) {
                @Override
                public IndexOutput createOutput(String name, IOContext context) throws IOException {
                    if (name.endsWith(KNN1040HalfFloatFlatVectorsFormat.VECTOR_DATA_EXTENSION)) {
                        throw new IOException("simulated failure creating vector data output");
                    }
                    return super.createOutput(name, context);
                }
            };

            SegmentInfo segmentInfo = createSegmentInfo(failingDir, "_0");
            FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { createFieldInfo() });
            SegmentWriteState writeState = new SegmentWriteState(
                InfoStream.NO_OUTPUT,
                failingDir,
                segmentInfo,
                fieldInfos,
                null,
                IOContext.DEFAULT
            );
            FlatVectorsScorer scorer = Mockito.mock(FlatVectorsScorer.class);

            IOException e = expectThrows(IOException.class, () -> new KNN1040HalfFloatFlatVectorsWriter(writeState, scorer));
            assertEquals("simulated failure creating vector data output", e.getMessage());
        }
    }

    @SneakyThrows
    public void testConstructor_directoryFailsCreatingMeta_cleansUpAndPropagatesOriginalException() {
        try (Directory baseDir = new ByteBuffersDirectory()) {
            Directory failingDir = new FilterDirectory(baseDir) {
                @Override
                public IndexOutput createOutput(String name, IOContext context) throws IOException {
                    if (name.endsWith(KNN1040HalfFloatFlatVectorsFormat.META_EXTENSION)) {
                        throw new IOException("simulated failure creating meta output");
                    }
                    return super.createOutput(name, context);
                }
            };

            SegmentInfo segmentInfo = createSegmentInfo(failingDir, "_0");
            FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { createFieldInfo() });
            SegmentWriteState writeState = new SegmentWriteState(
                InfoStream.NO_OUTPUT,
                failingDir,
                segmentInfo,
                fieldInfos,
                null,
                IOContext.DEFAULT
            );
            FlatVectorsScorer scorer = Mockito.mock(FlatVectorsScorer.class);

            IOException e = expectThrows(IOException.class, () -> new KNN1040HalfFloatFlatVectorsWriter(writeState, scorer));
            assertEquals("simulated failure creating meta output", e.getMessage());
        }
    }

    @SneakyThrows
    public void testFlush_multipleFields_writesEachFieldWithCorrectAlignment() {
        try (Directory dir = new ByteBuffersDirectory()) {
            FieldInfo fieldInfo1 = createFieldInfo("field_one", 0);
            FieldInfo fieldInfo2 = createFieldInfo("field_two", 1);
            SegmentInfo segmentInfo = createSegmentInfo(dir, "_0");
            FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { fieldInfo1, fieldInfo2 });
            SegmentWriteState writeState = new SegmentWriteState(
                InfoStream.NO_OUTPUT,
                dir,
                segmentInfo,
                fieldInfos,
                null,
                IOContext.DEFAULT
            );
            FlatVectorsScorer scorer = Mockito.mock(FlatVectorsScorer.class);

            float[][] vectors1 = { generateVector(), generateVector() };
            float[][] vectors2 = { generateVector() };

            try (FlatVectorsWriter writer = new KNN1040HalfFloatFlatVectorsWriter(writeState, scorer)) {
                @SuppressWarnings("unchecked")
                FlatFieldVectorsWriter<float[]> fieldWriter1 = (FlatFieldVectorsWriter<float[]>) writer.addField(fieldInfo1);
                @SuppressWarnings("unchecked")
                FlatFieldVectorsWriter<float[]> fieldWriter2 = (FlatFieldVectorsWriter<float[]>) writer.addField(fieldInfo2);
                for (int i = 0; i < vectors1.length; i++) {
                    fieldWriter1.addValue(i, vectors1[i]);
                }
                for (int i = 0; i < vectors2.length; i++) {
                    fieldWriter2.addValue(i, vectors2[i]);
                }
                writer.flush(Math.max(vectors1.length, vectors2.length), null);
                writer.finish();
            }

            String vectorDataFileName = IndexFileNames.segmentFileName("_0", "", KNN1040HalfFloatFlatVectorsFormat.VECTOR_DATA_EXTENSION);
            try (IndexInput input = dir.openInput(vectorDataFileName, IOContext.DEFAULT)) {
                CodecUtil.checkIndexHeader(
                    input,
                    KNN1040HalfFloatFlatVectorsFormat.VECTOR_DATA_CODEC_NAME,
                    KNN1040HalfFloatFlatVectorsFormat.VERSION_CURRENT,
                    KNN1040HalfFloatFlatVectorsFormat.VERSION_CURRENT,
                    segmentInfo.getId(),
                    ""
                );
                int byteSize = DIMENSION * Short.BYTES;
                byte[] raw = new byte[byteSize];

                assertVectorsAtAlignedOffset(input, vectors1, raw, byteSize);
                assertVectorsAtAlignedOffset(input, vectors2, raw, byteSize);
            }
        }
    }

    private void assertVectorsAtAlignedOffset(IndexInput input, float[][] vectors, byte[] raw, int byteSize) throws IOException {
        final int vectorDataAlignment = 64;
        long aligned = ((input.getFilePointer() + vectorDataAlignment - 1) / vectorDataAlignment) * vectorDataAlignment;
        input.seek(aligned);
        for (float[] expected : vectors) {
            input.readBytes(raw, 0, byteSize);
            float[] decoded = KNNVectorAsCollectionOfHalfFloatsSerializer.INSTANCE.byteToFloatArray(new BytesRef(raw));
            for (int d = 0; d < DIMENSION; d++) {
                float expectedFp16 = Float.float16ToFloat(Float.floatToFloat16(expected[d]));
                assertEquals(expectedFp16, decoded[d], 0.0f);
            }
        }
    }

    @SneakyThrows
    public void testFieldWriter_ramBytesUsed_emptyFieldReturnsShallowSizeOnly() {
        try (Directory dir = new ByteBuffersDirectory()) {
            try (FlatVectorsWriter writer = newWriter(dir, "_0")) {
                @SuppressWarnings("unchecked")
                FlatFieldVectorsWriter<float[]> fieldWriter = (FlatFieldVectorsWriter<float[]>) writer.addField(createFieldInfo());
                long empty = fieldWriter.ramBytesUsed();
                assertTrue(empty > 0);

                fieldWriter.addValue(0, generateVector());
                assertTrue("adding a vector should grow past the empty-field shallow size", fieldWriter.ramBytesUsed() > empty);
            }
        }
    }

    /**
     * Exercises {@code mergeOneFlatVectorField}, which delegates to
     * {@code KnnVectorsWriter.MergedVectorValues.mergeFloatVectorValues} to combine several source
     * readers' vectors in final merged-segment doc order. The source readers are hand-built (mocked
     * {@link KnnVectorsReader}s backed by plain in-memory {@link FloatVectorValues}) rather than real
     * {@code KNN1040HalfFloatFlatVectorsReader} instances, since the reader side isn't part of this PR.
     */
    @SneakyThrows
    public void testMergeOneFlatVectorField_mergesVectorsFromMultipleSourceReadersInOrder() {
        try (Directory dir = new ByteBuffersDirectory()) {
            float[][] source1Vectors = { generateVector(), generateVector() };
            float[][] source2Vectors = { generateVector() };

            FieldInfo fieldInfo = createFieldInfo();
            // maxDoc is set here directly since SegmentInfo#setMaxDoc is package-private to Lucene -
            // the real merge machinery (MergeState's CodecReader-based constructor) calls it as it
            // tallies docs across source readers; the raw-arrays constructor used below does not.
            SegmentInfo segmentInfo = createSegmentInfo(dir, "_merged", source1Vectors.length + source2Vectors.length);
            FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { fieldInfo });
            SegmentWriteState writeState = new SegmentWriteState(
                InfoStream.NO_OUTPUT,
                dir,
                segmentInfo,
                fieldInfos,
                null,
                IOContext.DEFAULT
            );
            FlatVectorsScorer scorer = Mockito.mock(FlatVectorsScorer.class);

            KnnVectorsReader reader1 = Mockito.mock(KnnVectorsReader.class);
            Mockito.when(reader1.getFloatVectorValues(FIELD_NAME)).thenReturn(wrapAsFloatVectorValues(source1Vectors));
            KnnVectorsReader reader2 = Mockito.mock(KnnVectorsReader.class);
            Mockito.when(reader2.getFloatVectorValues(FIELD_NAME)).thenReturn(wrapAsFloatVectorValues(source2Vectors));

            // reader1's docs (0, 1) map straight through; reader2's single doc lands after them (doc 2)
            // in the merged segment - a stand-in for however the real doc-remapping worked out.
            MergeState.DocMap[] docMaps = { docID -> docID, docID -> source1Vectors.length + docID };

            MergeState mergeState = new MergeState(
                docMaps,
                segmentInfo,
                fieldInfos,
                null,
                null,
                null,
                null,
                new FieldInfos[] { fieldInfos, fieldInfos },
                new Bits[] { null, null },
                null,
                null,
                new KnnVectorsReader[] { reader1, reader2 },
                new int[] { source1Vectors.length, source2Vectors.length },
                InfoStream.NO_OUTPUT,
                null,
                false,
                null
            );

            try (FlatVectorsWriter writer = new KNN1040HalfFloatFlatVectorsWriter(writeState, scorer)) {
                writer.mergeOneFlatVectorField(fieldInfo, mergeState);
                writer.finish();
            }

            String vectorDataFileName = IndexFileNames.segmentFileName(
                "_merged",
                "",
                KNN1040HalfFloatFlatVectorsFormat.VECTOR_DATA_EXTENSION
            );
            try (IndexInput input = dir.openInput(vectorDataFileName, IOContext.DEFAULT)) {
                CodecUtil.checkIndexHeader(
                    input,
                    KNN1040HalfFloatFlatVectorsFormat.VECTOR_DATA_CODEC_NAME,
                    KNN1040HalfFloatFlatVectorsFormat.VERSION_CURRENT,
                    KNN1040HalfFloatFlatVectorsFormat.VERSION_CURRENT,
                    segmentInfo.getId(),
                    ""
                );
                int byteSize = DIMENSION * Short.BYTES;
                byte[] raw = new byte[byteSize];
                float[][] expectedInOrder = { source1Vectors[0], source1Vectors[1], source2Vectors[0] };
                assertVectorsAtAlignedOffset(input, expectedInOrder, raw, byteSize);
            }
        }
    }

    private FloatVectorValues wrapAsFloatVectorValues(float[][] vectors) {
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
            public VectorEncoding getEncoding() {
                return VectorEncoding.FLOAT32;
            }

            @Override
            public DocIndexIterator iterator() {
                return createDenseIterator();
            }
        };
    }

    private FlatVectorsWriter newWriter(Directory dir, String segmentName) throws Exception {
        return newWriter(dir, createSegmentInfo(dir, segmentName));
    }

    private FlatVectorsWriter newWriter(Directory dir, SegmentInfo segmentInfo) throws Exception {
        FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { createFieldInfo() });
        SegmentWriteState writeState = new SegmentWriteState(InfoStream.NO_OUTPUT, dir, segmentInfo, fieldInfos, null, IOContext.DEFAULT);
        FlatVectorsScorer scorer = Mockito.mock(FlatVectorsScorer.class);
        return new KNN1040HalfFloatFlatVectorsWriter(writeState, scorer);
    }

    private SegmentInfo createSegmentInfo(Directory dir, String segmentName) {
        return createSegmentInfo(dir, segmentName, -1);
    }

    private SegmentInfo createSegmentInfo(Directory dir, String segmentName, int maxDoc) {
        return new SegmentInfo(
            dir,
            Version.LATEST,
            Version.LATEST,
            segmentName,
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
        return createFieldInfo(VectorEncoding.FLOAT32, VectorSimilarityFunction.EUCLIDEAN);
    }

    private FieldInfo createFieldInfo(VectorEncoding encoding, VectorSimilarityFunction similarity) {
        return createFieldInfo(FIELD_NAME, 0, encoding, similarity);
    }

    private FieldInfo createFieldInfo(String name, int number) {
        return createFieldInfo(name, number, VectorEncoding.FLOAT32, VectorSimilarityFunction.EUCLIDEAN);
    }

    private FieldInfo createFieldInfo(String name, int number, VectorEncoding encoding, VectorSimilarityFunction similarity) {
        return new FieldInfo(
            name,
            number,
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
            encoding,
            similarity,
            false,
            false
        );
    }

    private float[] generateVector() {
        float[] vector = new float[DIMENSION];
        for (int d = 0; d < DIMENSION; d++) {
            vector[d] = (random().nextFloat() * 2 - 1) * 10;
        }
        return vector;
    }
}
