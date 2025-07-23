/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import lombok.SneakyThrows;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsReader;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsWriter;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.ByteVectorValues;
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
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.util.UnitTestCodec;
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

@Log4j2

public class KNN990FlatVectorsFormatTests extends KNNTestCase {
    private static final Codec TESTING_CODEC = new UnitTestCodec(() -> new KNN990FlatVectorsFormat());
    private static final String FLAT_VECTOR_FILE_EXT = ".vec";
    private static final String FAISS_ENGINE_FILE_EXT = ".faiss";
    private static final String LUCENE_ENGINE_FILE_EXT = ".vex";
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
    public void testReaderAndWriter_whenValidInput_thenSuccess() {
        final Lucene99FlatVectorsFormat mockedFlatVectorsFormat = Mockito.mock(Lucene99FlatVectorsFormat.class);

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

        final KNN990FlatVectorsFormat knn990FlatVectorsFormat = new KNN990FlatVectorsFormat(mockedFlatVectorsFormat);
        try (MockedStatic<CodecUtil> mockedStaticCodecUtil = Mockito.mockStatic(CodecUtil.class)) {
            mockedStaticCodecUtil.when(
                () -> CodecUtil.writeIndexHeader(any(IndexOutput.class), anyString(), anyInt(), any(byte[].class), anyString())
            ).thenAnswer((Answer<Void>) invocation -> null);
            mockedStaticCodecUtil.when(() -> CodecUtil.retrieveChecksum(any(IndexInput.class)))
                .thenAnswer((Answer<Void>) invocation -> null);
            Assert.assertTrue(knn990FlatVectorsFormat.fieldsReader(mockedSegmentReadState) instanceof Lucene99FlatVectorsReader);

            Assert.assertTrue(knn990FlatVectorsFormat.fieldsWriter(mockedSegmentWriteState) instanceof Lucene99FlatVectorsWriter);
        }
    }

    @SneakyThrows
    public void testFlatVectorFormat_whenMultipleFloatVectorFieldIndexed_thenSuccess() {
        setup();
        float[] floatVector = { 1.0f, 3.0f, 4.0f };

        FieldType fieldTypeForFloat_Faiss = createVectorField(3, VectorEncoding.FLOAT32, VectorDataType.FLOAT, "faiss");
        fieldTypeForFloat_Faiss.freeze();
        addFieldToIndex(new KnnFloatVectorField("faiss_float", floatVector, fieldTypeForFloat_Faiss), indexWriter);
        FieldType fieldTypeForFloat_Lucene = createVectorField(3, VectorEncoding.FLOAT32, VectorDataType.FLOAT, "lucene");
        fieldTypeForFloat_Lucene.freeze();
        addFieldToIndex(new KnnFloatVectorField("lucene_float", floatVector, fieldTypeForFloat_Lucene), indexWriter);

        final IndexReader indexReader = indexWriter.getReader();
        // ensuring segments are created
        indexWriter.flush();
        indexWriter.commit();
        indexWriter.close();

        // Validate to see if correct values are returned, assumption here is only 1 segment is getting created
        IndexSearcher searcher = new IndexSearcher(indexReader);
        final LeafReader leafReader = searcher.getLeafContexts().get(0).reader();
        SegmentReader segmentReader = Lucene.segmentReader(leafReader);
        final List<String> faissfiles = getFilesFromSegment(dir, FAISS_ENGINE_FILE_EXT);
        final List<String> lucenefiles = getFilesFromSegment(dir, LUCENE_ENGINE_FILE_EXT);
        assertEquals(0, faissfiles.size());
        assertEquals(0, lucenefiles.size());

        // Even setting IWC to not use compound file it still uses compound file, hence ensuring we don't check .vec
        // file in case segment uses compound format. use this seed once we fix this to validate everything is
        // working or not. -Dtests.seed=CAAE1B8D573EEB7E
        if (segmentReader.getSegmentInfo().info.getUseCompoundFile() == false) {
            final List<String> vecfiles = getFilesFromSegment(dir, FLAT_VECTOR_FILE_EXT);
            assertEquals(2, vecfiles.size());
        }

        final FloatVectorValues floatVectorValuesFaiss = leafReader.getFloatVectorValues("faiss_float");
        floatVectorValuesFaiss.iterator().nextDoc();
        assertArrayEquals(floatVector, floatVectorValuesFaiss.vectorValue(floatVectorValuesFaiss.iterator().index()), 0.0f);
        assertEquals(1, floatVectorValuesFaiss.size());
        assertEquals(3, floatVectorValuesFaiss.dimension());

        final FloatVectorValues floatVectorValuesLucene = leafReader.getFloatVectorValues("lucene_float");
        floatVectorValuesLucene.iterator().nextDoc();
        assertArrayEquals(floatVector, floatVectorValuesLucene.vectorValue(floatVectorValuesLucene.iterator().index()), 0.0f);
        assertEquals(1, floatVectorValuesLucene.size());
        assertEquals(3, floatVectorValuesLucene.dimension());

        indexReader.close();
    }

    @SneakyThrows
    public void testFlatVectorFormat_whenMultipleByteVectorFieldIndexed_thenSuccess() {
        setup();
        byte[] byteVector = { 6, 14 };

        FieldType fieldTypeForByte_Faiss = createVectorField(2, VectorEncoding.BYTE, VectorDataType.BINARY, "faiss");
        fieldTypeForByte_Faiss.freeze();
        addFieldToIndex(new KnnByteVectorField("faiss_byte", byteVector, fieldTypeForByte_Faiss), indexWriter);
        FieldType fieldTypeForByte_Lucene = createVectorField(2, VectorEncoding.BYTE, VectorDataType.BINARY, "lucene");
        fieldTypeForByte_Lucene.freeze();
        addFieldToIndex(new KnnByteVectorField("lucene_byte", byteVector, fieldTypeForByte_Lucene), indexWriter);

        final IndexReader indexReader = indexWriter.getReader();
        // ensuring segments are created
        indexWriter.flush();
        indexWriter.commit();
        indexWriter.close();

        // Validate to see if correct values are returned, assumption here is only 1 segment is getting created
        IndexSearcher searcher = new IndexSearcher(indexReader);
        final LeafReader leafReader = searcher.getLeafContexts().get(0).reader();
        SegmentReader segmentReader = Lucene.segmentReader(leafReader);
        final List<String> faissfiles = getFilesFromSegment(dir, FAISS_ENGINE_FILE_EXT);
        final List<String> lucenefiles = getFilesFromSegment(dir, LUCENE_ENGINE_FILE_EXT);
        assertEquals(0, faissfiles.size());
        assertEquals(0, lucenefiles.size());

        // Even setting IWC to not use compound file it still uses compound file, hence ensuring we don't check .vec
        // file in case segment uses compound format. use this seed once we fix this to validate everything is
        // working or not. -Dtests.seed=CAAE1B8D573EEB7E
        if (segmentReader.getSegmentInfo().info.getUseCompoundFile() == false) {
            final List<String> vecfiles = getFilesFromSegment(dir, FLAT_VECTOR_FILE_EXT);
            assertEquals(2, vecfiles.size());
        }

        final ByteVectorValues byteVectorValuesFaiss = leafReader.getByteVectorValues("faiss_byte");
        byteVectorValuesFaiss.iterator().nextDoc();
        assertArrayEquals(byteVector, byteVectorValuesFaiss.vectorValue(byteVectorValuesFaiss.iterator().index()));
        assertEquals(1, byteVectorValuesFaiss.size());
        assertEquals(2, byteVectorValuesFaiss.dimension());

        final ByteVectorValues byteVectorValuesLucene = leafReader.getByteVectorValues("lucene_byte");
        byteVectorValuesLucene.iterator().nextDoc();
        assertArrayEquals(byteVector, byteVectorValuesLucene.vectorValue(byteVectorValuesLucene.iterator().index()));
        assertEquals(1, byteVectorValuesLucene.size());
        assertEquals(2, byteVectorValuesLucene.dimension());

        indexReader.close();
    }

    private void setup() throws IOException {
        dir = newFSDirectory(createTempDir());
        // on the mock directory Lucene goes ahead and does a search on different fields. We want to avoid that as of
        // now. Given we have not implemented search for the native engine format using codec, the dir.close fails
        // with exception. Hence, marking this as false.
        ((BaseDirectoryWrapper) dir).setCheckIndexOnClose(false);
        indexWriter = createIndexWriter(dir, null);
    }

    private RandomIndexWriter createIndexWriter(final Directory dir, String searchMode) throws IOException {
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

    private FieldType createVectorField(int dimension, VectorEncoding vectorEncoding, VectorDataType vectorDataType, String engine) {
        FieldType flatVectorField = new FieldType();
        flatVectorField.setTokenized(false);
        flatVectorField.setIndexOptions(IndexOptions.NONE);
        flatVectorField.putAttribute(KNNVectorFieldMapper.KNN_FIELD, "true");
        flatVectorField.putAttribute(KNNConstants.KNN_METHOD, KNNConstants.METHOD_HNSW);
        flatVectorField.putAttribute(KNNConstants.KNN_ENGINE, engine);
        flatVectorField.putAttribute(KNNConstants.VECTOR_DATA_TYPE_FIELD, vectorDataType.getValue());
        flatVectorField.setVectorAttributes(
            dimension,
            vectorEncoding,
            SpaceType.L2.getKnnVectorSimilarityFunction().getVectorSimilarityFunction()
        );
        return flatVectorField;
    }

    private List<String> getFilesFromSegment(Directory dir, String fileFormat) throws IOException {
        return Arrays.stream(dir.listAll()).filter(x -> x.contains(fileFormat)).collect(Collectors.toList());
    }
}
