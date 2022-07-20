/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import com.google.common.collect.ImmutableList;
import org.opensearch.common.Strings;
import org.opensearch.common.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.util.KNNEngine;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

public class LuceneIT extends KNNRestTestCase {

    private static final int DIMENSION = 3;
    private static final String DOC_ID = "doc1";
    private static final int EF_CONSTRUCTION = 128;
    private static final String INDEX_NAME = "test-index-1";
    private static final String FIELD_NAME = "test-field-1";
    private static final int M = 16;

    public void test_addDoc() throws IOException {
        List<Integer> mValues = ImmutableList.of(16, 32, 64, 128);
        List<Integer> efConstructionValues = ImmutableList.of(16, 32, 64, 128);

        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .startObject(KNNConstants.KNN_METHOD)
            .field(KNNConstants.NAME, KNNEngine.LUCENE.getMethod(KNNConstants.METHOD_HNSW).getMethodComponent().getName())
            .field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .field(KNNConstants.KNN_ENGINE, KNNEngine.LUCENE.getName())
            .startObject(KNNConstants.PARAMETERS)
            .field(KNNConstants.METHOD_PARAMETER_M, mValues.get(random().nextInt(mValues.size())))
            .field(KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION, efConstructionValues.get(random().nextInt(efConstructionValues.size())))
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        Map<String, Object> mappingMap = xContentBuilderToMap(builder);
        String mapping = Strings.toString(builder);

        createKnnIndex(INDEX_NAME, mapping);
        assertEquals(new TreeMap<>(mappingMap), new TreeMap<>(getIndexMappingAsMap(INDEX_NAME)));

        Float[] vector = new Float[] { 2.0f, 4.5f, 6.5f };
        addKnnDoc(INDEX_NAME, DOC_ID, FIELD_NAME, vector);

        refreshAllIndices();
        assertEquals(1, getDocCount(INDEX_NAME));

        deleteKNNIndex(INDEX_NAME);
    }

    public void test_updateDoc() throws Exception {
        createKnnIndexMappingWithLuceneEngine(2);
        Float[] vector = { 6.0f, 6.0f };
        addKnnDoc(INDEX_NAME, DOC_ID, FIELD_NAME, vector);

        Float[] updatedVector = { 8.0f, 8.0f };
        updateKnnDoc(INDEX_NAME, DOC_ID, FIELD_NAME, updatedVector);

        refreshAllIndices();
        assertEquals(1, getDocCount(INDEX_NAME));

        deleteKNNIndex(INDEX_NAME);
    }

    public void test_deleteDoc() throws Exception {
        createKnnIndexMappingWithLuceneEngine(2);
        Float[] vector = { 6.0f, 6.0f };
        addKnnDoc(INDEX_NAME, DOC_ID, FIELD_NAME, vector);

        deleteKnnDoc(INDEX_NAME, DOC_ID);

        refreshAllIndices();
        assertEquals(0, getDocCount(INDEX_NAME));

        deleteKNNIndex(INDEX_NAME);
    }

    private void createKnnIndexMappingWithLuceneEngine(int dimension) throws Exception {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .startObject(KNNConstants.KNN_METHOD)
            .field(KNNConstants.NAME, KNNEngine.LUCENE.getMethod(KNNConstants.METHOD_HNSW).getMethodComponent().getName())
            .field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .field(KNNConstants.KNN_ENGINE, KNNEngine.LUCENE.getName())
            .startObject(KNNConstants.PARAMETERS)
            .field(KNNConstants.METHOD_PARAMETER_M, M)
            .field(KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION, EF_CONSTRUCTION)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        String mapping = Strings.toString(builder);
        createKnnIndex(INDEX_NAME, mapping);
    }
}
