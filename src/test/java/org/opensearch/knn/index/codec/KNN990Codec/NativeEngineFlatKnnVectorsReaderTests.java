/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import com.google.common.collect.ImmutableMap;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.opensearch.Version;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.KNNCodecTestUtil;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexWriter;
import org.opensearch.knn.index.codec.util.KNNCodecUtil;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;
import org.opensearch.knn.index.vectorvalues.TestVectorValues;
import org.opensearch.knn.plugin.stats.KNNGraphValue;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import static org.opensearch.knn.common.FieldInfoExtractor.extractVectorDataType;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_SEARCH;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_M;
import static org.opensearch.knn.index.codec.KNNCodecTestUtil.assertFileInCorrectLocation;
import static org.opensearch.knn.index.codec.KNNCodecTestUtil.assertLoadableByEngine;
import static org.opensearch.knn.index.codec.KNNCodecTestUtil.assertValidFooter;

public class NativeEngineFlatKnnVectorsReaderTests extends KNNTestCase {
    private static final int EF_SEARCH = 10;
    private static final Map<String, ?> HNSW_METHODPARAMETERS = Map.of(METHOD_PARAMETER_EF_SEARCH, EF_SEARCH);

    private static Directory directory;
    private static Codec codec;

    @BeforeClass
    public static void setStaticVariables() {
        directory = newFSDirectory(createTempDir());
        codec = new KNN990Codec();
    }

    @AfterClass
    public static void closeStaticVariables() throws IOException {
        directory.close();
    }

    public void testAddKNNFloatField_FaissEngine_ReadFaissFile_success() throws IOException {
        String segmentName = "_0";
        int docsInSegment = 100;
        String fieldName = String.format("testField%s", randomAlphaOfLength(4));

        KNNEngine knnEngine = KNNEngine.FAISS;
        SpaceType spaceType = SpaceType.INNER_PRODUCT;
        int dimension = 16;

        SegmentInfo segmentInfo = KNNCodecTestUtil.segmentInfoBuilder()
                .directory(directory)
                .segmentName(segmentName)
                .docsInSegment(docsInSegment)
                .codec(codec)
                .build();
        KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder()
                .vectorDataType(VectorDataType.FLOAT)
                .versionCreated(Version.CURRENT)
                .build();
        KNNMethodContext knnMethodContext = new KNNMethodContext(
                knnEngine,
                spaceType,
                new MethodComponentContext(METHOD_HNSW, ImmutableMap.of(METHOD_PARAMETER_M, 16, METHOD_PARAMETER_EF_CONSTRUCTION, 512))
        );

        String parameterString = XContentFactory.jsonBuilder()
                .map(knnEngine.getKNNLibraryIndexingContext(knnMethodContext, knnMethodConfigContext).getLibraryParameters())
                .toString();

        FieldInfo[] fieldInfoArray = new FieldInfo[] {
                KNNCodecTestUtil.FieldInfoBuilder.builder(fieldName)
                        .addAttribute(KNNVectorFieldMapper.KNN_FIELD, "true")
                        .addAttribute(KNNConstants.KNN_ENGINE, knnEngine.getName())
                        .addAttribute(KNNConstants.SPACE_TYPE, spaceType.getValue())
                        .addAttribute(KNNConstants.PARAMETERS, parameterString)
                        .build() };


        FieldInfos fieldInfos = new FieldInfos(fieldInfoArray);
        SegmentWriteState state = new SegmentWriteState(null, directory, segmentInfo, fieldInfos, null, IOContext.DEFAULT);

        long initialRefreshOperations = KNNGraphValue.REFRESH_TOTAL_OPERATIONS.getValue();

        // Add documents to the field
        float[][] vectorsData = TestVectorValues.getRandomVectors(docsInSegment, dimension);
        List<float[]> vectorList = new ArrayList<>();
        for(int i = 0; i < docsInSegment; i++) {
            vectorList.add(vectorsData[i]);
        }
        TestVectorValues.PreDefinedFloatVectorValues preDefinedFloatVectorValues = new TestVectorValues.PreDefinedFloatVectorValues(vectorList);

        FieldInfo field = fieldInfoArray[0];
        final VectorDataType vectorDataType = extractVectorDataType(field);
        final KNNVectorValues<?> knnVectorValues = KNNVectorValuesFactory.getVectorValues(vectorDataType, preDefinedFloatVectorValues);

        NativeIndexWriter.getWriter(field, state).flushIndex(knnVectorValues, (int) knnVectorValues.totalLiveDocs());

        // The document should be created in the correct location
        String expectedFile = KNNCodecUtil.buildEngineFileName(segmentName, knnEngine.getVersion(), fieldName, knnEngine.getExtension());
        assertFileInCorrectLocation(state, expectedFile);

        // The footer should be valid
        assertValidFooter(state.directory, expectedFile);

        // The document should be readable by faiss
        assertLoadableByEngine(HNSW_METHODPARAMETERS, state, expectedFile, knnEngine, spaceType, dimension);

        // The graph creation statistics should be updated
        assertEquals(1 + initialRefreshOperations, (long) KNNGraphValue.REFRESH_TOTAL_OPERATIONS.getValue());

        // Files Should set into segment info
        segmentInfo.setFiles(Collections.singleton(expectedFile));

        // Reader From Faiss File and get FloatVectorValues
        SegmentReadState readState = new SegmentReadState(directory, segmentInfo, fieldInfos, IOContext.DEFAULT);
        FaissEngineFlatKnnVectorsReader faissReader = new FaissEngineFlatKnnVectorsReader(readState);
        FloatVectorValues vectorValues = faissReader.getFloatVectorValues(fieldName);

        FaissEngineFlatKnnVectorsReader.MetaInfo metaInfo = faissReader.getFieldMetaMap().get(fieldName);

        for (int i = 0; i < metaInfo.ntotal; i++){
            vectorValues.nextDoc();
            float[] actualVector = vectorValues.vectorValue();
            float[] expectVector = vectorsData[i];
            assertArrayEquals(actualVector, expectVector, 0.001f);
        }
        faissReader.close();
    }
}
