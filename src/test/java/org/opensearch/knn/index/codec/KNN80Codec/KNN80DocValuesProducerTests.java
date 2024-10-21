/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN80Codec;

import com.google.common.collect.ImmutableMap;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.DocValuesFormat;
import org.apache.lucene.codecs.DocValuesProducer;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexOutput;
import org.junit.Before;
import org.opensearch.Version;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.KNN87Codec.KNN87Codec;
import org.opensearch.knn.index.codec.KNNCodecTestUtil;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_M;

public class KNN80DocValuesProducerTests extends KNNTestCase {

    private static Directory directory;

    @Before
    public void setUp() throws Exception {
        super.setUp();
        directory = newFSDirectory(createTempDir());
    }

    @Override
    public void tearDown() throws Exception {
        super.tearDown();
        directory.close();
    }

    public void testProduceKNNBinaryField_fromCodec_nmslibCurrent() throws IOException {
        // Set information about the segment and the fields
        DocValuesFormat mockDocValuesFormat = mock(DocValuesFormat.class);
        Codec mockDelegateCodec = mock(Codec.class);
        DocValuesProducer mockDocValuesProducer = mock(DocValuesProducer.class);
        when(mockDelegateCodec.docValuesFormat()).thenReturn(mockDocValuesFormat);
        when(mockDocValuesFormat.fieldsProducer(any())).thenReturn(mockDocValuesProducer);
        when(mockDocValuesFormat.getName()).thenReturn("mockDocValuesFormat");
        Codec codec = new KNN87Codec(mockDelegateCodec);

        String segmentName = "_test";
        int docsInSegment = 100;
        String fieldName1 = String.format("test_field1%s", randomAlphaOfLength(4));
        String fieldName2 = String.format("test_field2%s", randomAlphaOfLength(4));
        List<String> segmentFiles = Arrays.asList(
            String.format("%s_2011_%s%s", segmentName, fieldName1, KNNEngine.NMSLIB.getExtension()),
            String.format("%s_165_%s%s", segmentName, fieldName2, KNNEngine.FAISS.getExtension())
        );

        KNNEngine knnEngine = KNNEngine.NMSLIB;
        SpaceType spaceType = SpaceType.COSINESIMIL;
        SegmentInfo segmentInfo = KNNCodecTestUtil.segmentInfoBuilder()
            .directory(directory)
            .segmentName(segmentName)
            .docsInSegment(docsInSegment)
            .codec(codec)
            .build();

        for (String name : segmentFiles) {
            IndexOutput indexOutput = directory.createOutput(name, IOContext.DEFAULT);
            indexOutput.close();
        }
        segmentInfo.setFiles(segmentFiles);

        KNNMethodContext knnMethodContext = new KNNMethodContext(
            knnEngine,
            spaceType,
            new MethodComponentContext(METHOD_HNSW, ImmutableMap.of(METHOD_PARAMETER_M, 16, METHOD_PARAMETER_EF_CONSTRUCTION, 512))
        );
        KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder()
            .vectorDataType(VectorDataType.FLOAT)
            .versionCreated(Version.CURRENT)
            .build();
        String parameterString = XContentFactory.jsonBuilder()
            .map(knnEngine.getKNNLibraryIndexingContext(knnMethodContext, knnMethodConfigContext).getLibraryParameters())
            .toString();

        FieldInfo[] fieldInfoArray = new FieldInfo[] {
            KNNCodecTestUtil.FieldInfoBuilder.builder(fieldName1)
                .addAttribute(KNNVectorFieldMapper.KNN_FIELD, "true")
                .addAttribute(KNNConstants.KNN_ENGINE, knnEngine.getName())
                .addAttribute(KNNConstants.SPACE_TYPE, spaceType.getValue())
                .addAttribute(KNNConstants.PARAMETERS, parameterString)
                .build() };

        FieldInfos fieldInfos = new FieldInfos(fieldInfoArray);
        SegmentReadState state = new SegmentReadState(directory, segmentInfo, fieldInfos, IOContext.DEFAULT);

        DocValuesFormat docValuesFormat = codec.docValuesFormat();
        assertTrue(docValuesFormat instanceof KNN80DocValuesFormat);
        DocValuesProducer producer = docValuesFormat.fieldsProducer(state);
        assertTrue(producer instanceof KNN80DocValuesProducer);
        int cacheKeySize = ((KNN80DocValuesProducer) producer).getCacheKeys().size();
        assertEquals(cacheKeySize, 1);

        String cacheKey = ((KNN80DocValuesProducer) producer).getCacheKeys().get(0);
        assertTrue(cacheKey.contains(segmentFiles.get(0)));
    }

    public void testProduceKNNBinaryField_whenFieldHasNonBinaryDocValues_thenSkipThoseField() throws IOException {
        // Set information about the segment and the fields
        DocValuesFormat mockDocValuesFormat = mock(DocValuesFormat.class);
        Codec mockDelegateCodec = mock(Codec.class);
        DocValuesProducer mockDocValuesProducer = mock(DocValuesProducer.class);
        when(mockDelegateCodec.docValuesFormat()).thenReturn(mockDocValuesFormat);
        when(mockDocValuesFormat.fieldsProducer(any())).thenReturn(mockDocValuesProducer);
        when(mockDocValuesFormat.getName()).thenReturn("mockDocValuesFormat");
        Codec codec = new KNN87Codec(mockDelegateCodec);

        String segmentName = "_test";
        int docsInSegment = 100;
        String fieldName1 = String.format("test_field1%s", randomAlphaOfLength(4));
        String fieldName2 = String.format("test_field2%s", randomAlphaOfLength(4));
        List<String> segmentFiles = Arrays.asList(
            String.format("%s_2011_%s%s", segmentName, fieldName1, KNNEngine.NMSLIB.getExtension()),
            String.format("%s_165_%s%s", segmentName, fieldName2, KNNEngine.FAISS.getExtension())
        );

        KNNEngine knnEngine = KNNEngine.NMSLIB;
        SpaceType spaceType = SpaceType.COSINESIMIL;
        SegmentInfo segmentInfo = KNNCodecTestUtil.segmentInfoBuilder()
            .directory(directory)
            .segmentName(segmentName)
            .docsInSegment(docsInSegment)
            .codec(codec)
            .build();

        for (String name : segmentFiles) {
            IndexOutput indexOutput = directory.createOutput(name, IOContext.DEFAULT);
            indexOutput.close();
        }
        segmentInfo.setFiles(segmentFiles);

        FieldInfo[] fieldInfoArray = new FieldInfo[] {
            KNNCodecTestUtil.FieldInfoBuilder.builder(fieldName1)
                .addAttribute(KNNVectorFieldMapper.KNN_FIELD, "true")
                .addAttribute(KNNConstants.KNN_ENGINE, knnEngine.getName())
                .addAttribute(KNNConstants.SPACE_TYPE, spaceType.getValue())
                .docValuesType(DocValuesType.NONE)
                .dvGen(-1)
                .build() };

        FieldInfos fieldInfos = new FieldInfos(fieldInfoArray);
        SegmentReadState state = new SegmentReadState(directory, segmentInfo, fieldInfos, IOContext.DEFAULT);

        DocValuesFormat docValuesFormat = codec.docValuesFormat();
        assertTrue(docValuesFormat instanceof KNN80DocValuesFormat);
        DocValuesProducer producer = docValuesFormat.fieldsProducer(state);
        assertTrue(producer instanceof KNN80DocValuesProducer);
        assertEquals(0, ((KNN80DocValuesProducer) producer).getCacheKeys().size());
    }

}
