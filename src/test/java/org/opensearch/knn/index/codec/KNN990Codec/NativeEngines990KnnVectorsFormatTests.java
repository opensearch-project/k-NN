/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import lombok.SneakyThrows;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.SerialMergeScheduler;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.index.RandomIndexWriter;
import org.apache.lucene.tests.store.BaseDirectoryWrapper;
import org.apache.lucene.util.Bits;
import org.junit.After;
import org.junit.Assert;
import org.mockito.Mockito;
import org.opensearch.common.lucene.Lucene;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.index.util.KNNEngine;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

@Log4j2
public class NativeEngines990KnnVectorsFormatTests extends KNNTestCase {
    private static final Codec TESTING_CODEC = new UnitTestCodec();
    private static final String FLAT_VECTOR_FILE_EXT = ".vec";
    private static final String HNSW_FILE_EXT = ".hnsw";
    private static final String FLOAT_VECTOR_FIELD = "float_field";
    private static final String BYTE_VECTOR_FIELD = "byte_field";
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
        final SegmentWriteState mockedSegmentWriteState = Mockito.mock(SegmentWriteState.class);
        final SegmentReadState mockedSegmentReadState = Mockito.mock(SegmentReadState.class);

        Mockito.when(mockedFlatVectorsFormat.fieldsReader(mockedSegmentReadState)).thenReturn(Mockito.mock(FlatVectorsReader.class));
        Mockito.when(mockedFlatVectorsFormat.fieldsWriter(mockedSegmentWriteState)).thenReturn(Mockito.mock(FlatVectorsWriter.class));

        final NativeEngines990KnnVectorsFormat nativeEngines990KnnVectorsFormat = new NativeEngines990KnnVectorsFormat(
            mockedFlatVectorsFormat
        );
        Assert.assertTrue(
            nativeEngines990KnnVectorsFormat.fieldsReader(mockedSegmentReadState) instanceof NativeEngines990KnnVectorsReader
        );
        Assert.assertTrue(
            nativeEngines990KnnVectorsFormat.fieldsWriter(mockedSegmentWriteState) instanceof NativeEngines990KnnVectorsWriter
        );
    }

    @SneakyThrows
    public void testNativeEngineVectorFormat_whenMultipleVectorFieldIndexed_thenSuccess() {
        setup();
        float[] floatVector = { 1.0f, 3.0f, 4.0f };
        byte[] byteVector = { 6, 14 };

        addFieldToIndex(
            new KnnFloatVectorField(FLOAT_VECTOR_FIELD, floatVector, createVectorField(3, VectorEncoding.FLOAT32)),
            indexWriter
        );
        addFieldToIndex(new KnnByteVectorField(BYTE_VECTOR_FIELD, byteVector, createVectorField(2, VectorEncoding.BYTE)), indexWriter);
        final IndexReader indexReader = indexWriter.getReader();
        // ensuring segments are created
        indexWriter.flush();
        indexWriter.commit();
        indexWriter.close();

        // Validate to see if correct values are returned, assumption here is only 1 segment is getting created
        IndexSearcher searcher = new IndexSearcher(indexReader);
        final LeafReader leafReader = searcher.getLeafContexts().get(0).reader();
        SegmentReader segmentReader = Lucene.segmentReader(leafReader);
        final List<String> hnswfiles = getFilesFromSegment(dir, HNSW_FILE_EXT);
        // 0 hnsw files for now as we have not integrated graph creation here.
        assertEquals(0, hnswfiles.size());
        assertEquals(hnswfiles.stream().filter(x -> x.contains(FLOAT_VECTOR_FIELD)).count(), 0);
        assertEquals(hnswfiles.stream().filter(x -> x.contains(BYTE_VECTOR_FIELD)).count(), 0);

        // Even setting IWC to not use compound file it still uses compound file, hence ensuring we don't check .vec
        // file in case segment uses compound format. use this seed once we fix this to validate everything is
        // working or not. -Dtests.seed=CAAE1B8D573EEB7E
        if (segmentReader.getSegmentInfo().info.getUseCompoundFile() == false) {
            final List<String> vecfiles = getFilesFromSegment(dir, FLAT_VECTOR_FILE_EXT);
            // 2 .vec files will be created as we are using per field vectors format.
            assertEquals(2, vecfiles.size());
        }

        final FloatVectorValues floatVectorValues = leafReader.getFloatVectorValues(FLOAT_VECTOR_FIELD);
        floatVectorValues.nextDoc();
        assertArrayEquals(floatVector, floatVectorValues.vectorValue(), 0.0f);
        assertEquals(1, floatVectorValues.size());
        assertEquals(3, floatVectorValues.dimension());

        final ByteVectorValues byteVectorValues = leafReader.getByteVectorValues(BYTE_VECTOR_FIELD);
        byteVectorValues.nextDoc();
        assertArrayEquals(byteVector, byteVectorValues.vectorValue());
        assertEquals(1, byteVectorValues.size());
        assertEquals(2, byteVectorValues.dimension());

        Assert.assertThrows(
            UnsupportedOperationException.class,
            () -> leafReader.searchNearestVectors(FLOAT_VECTOR_FIELD, floatVector, 10, new Bits.MatchAllBits(1), 10)
        );

        Assert.assertThrows(
            UnsupportedOperationException.class,
            () -> leafReader.searchNearestVectors(BYTE_VECTOR_FIELD, byteVector, 10, new Bits.MatchAllBits(1), 10)
        );
        // do it at the end so that all search is completed
        indexReader.close();
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
        FieldType nativeVectorField = new FieldType();
        // TODO: Replace this with the default field which will be created in mapper for Native Engines with KNNVectorsFormat
        nativeVectorField.setTokenized(false);
        nativeVectorField.setIndexOptions(IndexOptions.NONE);
        nativeVectorField.putAttribute(KNNVectorFieldMapper.KNN_FIELD, "true");
        nativeVectorField.putAttribute(KNNConstants.KNN_METHOD, KNNConstants.METHOD_HNSW);
        nativeVectorField.putAttribute(KNNConstants.KNN_ENGINE, KNNEngine.NMSLIB.getName());
        nativeVectorField.putAttribute(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue());
        nativeVectorField.putAttribute(KNNConstants.HNSW_ALGO_M, "32");
        nativeVectorField.putAttribute(KNNConstants.HNSW_ALGO_EF_CONSTRUCTION, "512");
        nativeVectorField.setVectorAttributes(
            dimension,
            vectorEncoding,
            SpaceType.L2.getKnnVectorSimilarityFunction().getVectorSimilarityFunction()
        );
        nativeVectorField.freeze();
        return nativeVectorField;
    }
}
