/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN990Codec.halffloatcodec;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.SerialMergeScheduler;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Sort;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.tests.index.RandomIndexWriter;
import org.apache.lucene.tests.store.BaseDirectoryWrapper;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.Version;

import org.junit.After;
import org.junit.Assert;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.mockito.stubbing.Answer;
import org.opensearch.common.lucene.Lucene;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.util.UnitTestCodec;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;

import java.io.IOException;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyString;

public class KNN990HalfFloatFlatVectorsFormatTests extends KNNTestCase {
    private static final Codec TESTING_CODEC = new UnitTestCodec(() -> new KNN990HalfFloatFlatVectorsFormat(new DefaultFlatVectorScorer()));
    private static final String FLAT_VECTOR_FILE_EXT = ".vec";
    private static final String FLOAT_VECTOR_FIELD = "float_field";
    private Directory dir;
    private RandomIndexWriter indexWriter;

    @After
    public void tearDown() throws Exception {
        if (dir != null) {
            dir.close();
        }
        super.tearDown();
    }

    @SneakyThrows
    public void testHalfFloatVectorFormat_whenMultipleVectorFieldIndexed_halfFloat_thenSuccess() {
        setup();
        float[] floatVector = { 1.0f, 3.0f, 4.0f };
        float[] halfFloatVector = { 2.0f, 4.0f, 8.0f };

        FieldType fieldTypeForFloat = createVectorField(floatVector.length, VectorEncoding.FLOAT32);
        fieldTypeForFloat.putAttribute(KNNConstants.PARAMETERS, "{ \"index_description\":\"HNSW16,Flat\", \"spaceType\": \"l2\"}");
        fieldTypeForFloat.freeze();
        addFieldToIndex(new KnnFloatVectorField(FLOAT_VECTOR_FIELD, floatVector, fieldTypeForFloat), indexWriter);

        FieldType fieldTypeForHalfFloat = createVectorField(halfFloatVector.length, VectorEncoding.FLOAT32);
        fieldTypeForHalfFloat.putAttribute(KNNConstants.PARAMETERS, "{ \"index_description\":\"HNSW16,Flat\", \"spaceType\": \"l2\"}");
        fieldTypeForHalfFloat.freeze();
        addFieldToIndex(new KnnFloatVectorField("half_float_field", halfFloatVector, fieldTypeForHalfFloat), indexWriter);

        final IndexReader indexReader = indexWriter.getReader();
        indexWriter.flush();
        indexWriter.commit();
        indexWriter.close();

        IndexSearcher searcher = new IndexSearcher(indexReader);
        final LeafReader leafReader = searcher.getLeafContexts().get(0).reader();
        SegmentReader segmentReader = Lucene.segmentReader(leafReader);

        if (segmentReader.getSegmentInfo().info.getUseCompoundFile() == false) {
            final List<String> vecfiles = getFilesFromSegment(dir, FLAT_VECTOR_FILE_EXT);
            assertEquals(2, vecfiles.size());
        }

        final FloatVectorValues floatVectorValues = leafReader.getFloatVectorValues(FLOAT_VECTOR_FIELD);
        floatVectorValues.iterator().nextDoc();
        assertArrayEquals(floatVector, floatVectorValues.vectorValue(floatVectorValues.iterator().index()), 0.0f);
        assertEquals(1, floatVectorValues.size());
        assertEquals(floatVector.length, floatVectorValues.dimension());

        final FloatVectorValues halfFloatVectorValues = leafReader.getFloatVectorValues("half_float_field");
        halfFloatVectorValues.iterator().nextDoc();
        assertArrayEquals(halfFloatVector, halfFloatVectorValues.vectorValue(halfFloatVectorValues.iterator().index()), 0.0f);
        assertEquals(1, halfFloatVectorValues.size());
        assertEquals(halfFloatVector.length, halfFloatVectorValues.dimension());

        indexReader.close();
    }

    public void testFormatName_withValidInput_thenSuccess() {
        final String validFormatName = "KNN990HalfFloatFlatVectorsFormat";
        assertEquals(validFormatName, new KNN990HalfFloatFlatVectorsFormat().getName());
        assertEquals(validFormatName, new KNN990HalfFloatFlatVectorsFormat(new DefaultFlatVectorScorer()).getName());
    }

    @SneakyThrows
    public void testReaderAndWriter_whenValidInput_thenSuccess() {
        final KNN990HalfFloatFlatVectorsFormat mockedFlatVectorsFormat = Mockito.mock(KNN990HalfFloatFlatVectorsFormat.class);

        final String segmentName = "test-segment-name";

        final SegmentInfo mockedSegmentInfo = new SegmentInfo(
            Mockito.mock(Directory.class),
            Mockito.mock(Version.class),
            Mockito.mock(Version.class),
            segmentName,
            0,
            false,
            false,
            Mockito.mock(Codec.class),
            Mockito.mock(Map.class),
            new byte[16],
            Mockito.mock(Map.class),
            Mockito.mock(Sort.class)
        );

        final String segmentSuffix = "test-segment-suffix";

        Directory directory = Mockito.mock(Directory.class);
        IndexInput input = Mockito.mock(IndexInput.class);
        Mockito.when(directory.openInput(any(), any())).thenReturn(input);

        String fieldName = "test-field";
        FieldInfos fieldInfos = Mockito.mock(FieldInfos.class);
        FieldInfo fieldInfo = Mockito.mock(FieldInfo.class);
        Mockito.when(fieldInfo.getName()).thenReturn(fieldName);
        Mockito.when(fieldInfos.fieldInfo(anyInt())).thenReturn(fieldInfo);
        Mockito.when(fieldInfos.iterator()).thenReturn(new Iterator<FieldInfo>() {
            @Override
            public boolean hasNext() {
                return false;
            }

            @Override
            public FieldInfo next() {
                return null;
            }
        });

        final SegmentReadState mockedSegmentReadState = new SegmentReadState(
            directory,
            mockedSegmentInfo,
            fieldInfos,
            Mockito.mock(IOContext.class),
            segmentSuffix
        );

        final SegmentWriteState mockedSegmentWriteState = new SegmentWriteState(
            Mockito.mock(InfoStream.class),
            Mockito.mock(Directory.class),
            mockedSegmentInfo,
            Mockito.mock(FieldInfos.class),
            null,
            Mockito.mock(IOContext.class)
        );
        Mockito.when(mockedFlatVectorsFormat.fieldsReader(mockedSegmentReadState)).thenReturn(Mockito.mock(FlatVectorsReader.class));
        Mockito.when(mockedFlatVectorsFormat.fieldsWriter(mockedSegmentWriteState)).thenReturn(Mockito.mock(FlatVectorsWriter.class));

        final KNN990HalfFloatFlatVectorsFormat halfFloatFlatVectorsFormat = new KNN990HalfFloatFlatVectorsFormat();

        try (MockedStatic<CodecUtil> mockedStaticCodecUtil = Mockito.mockStatic(CodecUtil.class)) {
            mockedStaticCodecUtil.when(
                () -> CodecUtil.writeIndexHeader(any(IndexOutput.class), anyString(), anyInt(), any(byte[].class), anyString())
            ).thenAnswer((Answer<Void>) invocation -> null);
            mockedStaticCodecUtil.when(() -> CodecUtil.retrieveChecksum(any(IndexInput.class)))
                .thenAnswer((Answer<Void>) invocation -> null);

            Object reader = halfFloatFlatVectorsFormat.fieldsReader(mockedSegmentReadState);
            Object writer = halfFloatFlatVectorsFormat.fieldsWriter(mockedSegmentWriteState);

            Assert.assertNotNull(reader);
            Assert.assertNotNull(writer);
        }
    }

    private List<String> getFilesFromSegment(Directory dir, String fileFormat) throws IOException {
        return Arrays.stream(dir.listAll()).filter(x -> x.contains(fileFormat)).collect(Collectors.toList());
    }

    /**
     * This should have been annotated with @Before, but somehow when I annotate with @Before apart from running
     * before tests, it is also running independently and failing. Need to figure this out.
     * @throws IOException
     */
    private void setup() throws IOException {
        dir = newFSDirectory(createTempDir());
        // on the mock directory Lucene goes ahead and does a search on different fields. We want to avoid that as of
        // now. Given we have not implemented search for the native engine format using codec, the dir.close fails
        // with exception. Hence, marking this as false.
        ((BaseDirectoryWrapper) dir).setCheckIndexOnClose(false);
        indexWriter = createIndexWriter(dir);
    }

    private RandomIndexWriter createIndexWriter(final Directory dir) throws IOException {
        final IndexWriterConfig iwc = newIndexWriterConfig();
        iwc.setMergeScheduler(new SerialMergeScheduler());
        iwc.setCodec(TESTING_CODEC);
        iwc.setUseCompoundFile(false);
        // Set merge policy to no merges so that we create a predictable number of segments.
        iwc.setMergePolicy(NoMergePolicy.INSTANCE);
        return new RandomIndexWriter(random(), dir, iwc);
    }

    private void addFieldToIndex(final Field vectorField, final RandomIndexWriter indexWriter) throws IOException {
        final Document doc1 = new Document();
        doc1.add(vectorField);
        indexWriter.addDocument(doc1);
    }

    private FieldType createVectorField(int dimension, VectorEncoding vectorEncoding) {
        FieldType halfFloatVectorField = new FieldType();
        halfFloatVectorField.setTokenized(false);
        halfFloatVectorField.setIndexOptions(IndexOptions.NONE);
        halfFloatVectorField.putAttribute(KNNVectorFieldMapper.KNN_FIELD, "true");
        halfFloatVectorField.putAttribute(KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD, "-1");
        halfFloatVectorField.putAttribute(KNNConstants.KNN_METHOD, KNNConstants.METHOD_HNSW);
        halfFloatVectorField.putAttribute(KNNConstants.KNN_ENGINE, KNNEngine.LUCENE.getName());
        halfFloatVectorField.putAttribute(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue());
        halfFloatVectorField.putAttribute(KNNConstants.HNSW_ALGO_M, "32");
        halfFloatVectorField.putAttribute(KNNConstants.HNSW_ALGO_EF_CONSTRUCTION, "512");
        halfFloatVectorField.putAttribute(KNNConstants.VECTOR_DATA_TYPE_FIELD, VectorDataType.HALF_FLOAT.getValue());
        halfFloatVectorField.setVectorAttributes(
            dimension,
            vectorEncoding,
            SpaceType.L2.getKnnVectorSimilarityFunction().getVectorSimilarityFunction()
        );
        return halfFloatVectorField;
    }
}
