/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import com.google.common.collect.ImmutableMap;
import lombok.SneakyThrows;
import lombok.extern.log4j.Log4j2;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Response;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.ToXContent;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNJsonIndexMappingsBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.query.KNNExactQueryBuilder;
import org.opensearch.knn.plugin.script.KNNScoringUtil;

import java.io.IOException;
import java.util.List;

import static org.opensearch.knn.common.KNNConstants.QUERY;
import static org.opensearch.knn.common.KNNConstants.EXACT_KNN;
import static org.opensearch.knn.common.KNNConstants.VECTOR;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;

@Log4j2
public class KNNExactQueryIT extends KNNRestTestCase {

    private static final float[] QUERY_VECTOR = { 1.0f, 2.0f, 3.0f };
    private static final byte[] BYTE_QUERY_VECTOR = { 1, 2, 3 };
    private static final int DIMENSION = 3;
    private static final int SIZE = 5;
    private static final String FIELD_NAME_NESTED = "nested_test";
    private static final String FIELD_NAME_FILTER = "color";

    @SneakyThrows
    public void testKNNExactQuery_Basic() {
        createTestIndex(false);
        for (int i = 0; i < SIZE; i++) {
            addKnnDoc(INDEX_NAME, String.valueOf(i), FIELD_NAME, new float[] { i, i, i });
        }

        KNNExactQueryBuilder exactQueryBuilder = KNNExactQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(QUERY_VECTOR)
            .spaceType("innerproduct")
            .build();

        Response searchResponse = validateKNNExactSearch(INDEX_NAME, exactQueryBuilder);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(searchResponse.getEntity()), FIELD_NAME);

        float[] expectedResults = new float[SIZE];
        for (int i = 0; i < SIZE; i++) {
            expectedResults[i] = SpaceType.INNER_PRODUCT.getKnnVectorSimilarityFunction().compare(QUERY_VECTOR, new float[] { i, i, i });
        }

        assertEquals(SIZE, results.size());
        for (int i = 0; i < SIZE; i++) {
            String docId = results.get(i).getDocId();
            assertEquals(expectedResults[Integer.parseInt(docId)], results.get(i).getScore(), 0.00001);
        }
        deleteKNNIndex(INDEX_NAME);
    }

    @SneakyThrows
    public void testKNNExactQuery_BasicLucene() {
        createTestIndex(true);
        for (int i = 0; i < SIZE; i++) {
            addKnnDoc(INDEX_NAME, String.valueOf(i), FIELD_NAME, new float[] { i, i, i });
        }

        KNNExactQueryBuilder exactQueryBuilder = KNNExactQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(QUERY_VECTOR)
            .spaceType("innerproduct")
            .build();

        Response searchResponse = validateKNNExactSearch(INDEX_NAME, exactQueryBuilder);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(searchResponse.getEntity()), FIELD_NAME);

        float[] expectedResults = new float[SIZE];
        for (int i = 0; i < SIZE; i++) {
            expectedResults[i] = SpaceType.INNER_PRODUCT.getKnnVectorSimilarityFunction().compare(QUERY_VECTOR, new float[] { i, i, i });
        }

        assertEquals(SIZE, results.size());
        for (int i = 0; i < SIZE; i++) {
            String docId = results.get(i).getDocId();
            assertEquals(expectedResults[Integer.parseInt(docId)], results.get(i).getScore(), 0.00001);
        }
        deleteKNNIndex(INDEX_NAME);
    }

    @SneakyThrows
    public void testKNNExactQuery_NoSpaceTypeDefined_ThenIndexSpaceTypeL2Used() {
        createTestIndex(false);
        for (int i = 0; i < SIZE; i++) {
            addKnnDoc(INDEX_NAME, String.valueOf(i), FIELD_NAME, new float[] { i, i, i });
        }

        KNNExactQueryBuilder exactQueryBuilder = KNNExactQueryBuilder.builder().fieldName(FIELD_NAME).vector(QUERY_VECTOR).build();

        Response searchResponse = validateKNNExactSearch(INDEX_NAME, exactQueryBuilder);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(searchResponse.getEntity()), FIELD_NAME);

        float[] expectedResults = new float[SIZE];
        for (int i = 0; i < SIZE; i++) {
            expectedResults[i] = SpaceType.L2.getKnnVectorSimilarityFunction().compare(QUERY_VECTOR, new float[] { i, i, i });
        }

        assertEquals(SIZE, results.size());
        for (int i = 0; i < SIZE; i++) {
            String docId = results.get(i).getDocId();
            assertEquals(expectedResults[Integer.parseInt(docId)], results.get(i).getScore(), 0.00001);
        }
        deleteKNNIndex(INDEX_NAME);
    }

    @SneakyThrows
    public void testKNNExactQuery_SpaceTypeL1() {
        createTestIndex(false);
        for (int i = 0; i < SIZE; i++) {
            addKnnDoc(INDEX_NAME, String.valueOf(i), FIELD_NAME, new float[] { i, i, i });
        }

        KNNExactQueryBuilder exactQueryBuilder = KNNExactQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(QUERY_VECTOR)
            .spaceType("l1")
            .build();

        Response searchResponse = validateKNNExactSearch(INDEX_NAME, exactQueryBuilder);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(searchResponse.getEntity()), FIELD_NAME);

        float[] expectedResults = new float[SIZE];
        for (int i = 0; i < SIZE; i++) {
            expectedResults[i] = 1 / (1 + KNNScoringUtil.l1Norm(QUERY_VECTOR, new float[] { i, i, i }));
        }

        assertEquals(SIZE, results.size());
        for (int i = 0; i < SIZE; i++) {
            String docId = results.get(i).getDocId();
            assertEquals(expectedResults[Integer.parseInt(docId)], results.get(i).getScore(), 0.00001);
        }
        deleteKNNIndex(INDEX_NAME);
    }

    @SneakyThrows
    public void testKNNExactQuery_Lucene_SpaceTypeLINF() {
        createTestIndex(true);
        for (int i = 0; i < SIZE; i++) {
            addKnnDoc(INDEX_NAME, String.valueOf(i), FIELD_NAME, new float[] { i, i, i });
        }

        KNNExactQueryBuilder exactQueryBuilder = KNNExactQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(QUERY_VECTOR)
            .spaceType("linf")
            .build();

        Response searchResponse = validateKNNExactSearch(INDEX_NAME, exactQueryBuilder);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(searchResponse.getEntity()), FIELD_NAME);

        float[] expectedResults = new float[SIZE];
        for (int i = 0; i < SIZE; i++) {
            expectedResults[i] = 1 / (1 + KNNScoringUtil.lInfNorm(QUERY_VECTOR, new float[] { i, i, i }));
        }

        assertEquals(SIZE, results.size());
        for (int i = 0; i < SIZE; i++) {
            String docId = results.get(i).getDocId();
            assertEquals(expectedResults[Integer.parseInt(docId)], results.get(i).getScore(), 0.00001);
        }
        deleteKNNIndex(INDEX_NAME);
    }

    @SneakyThrows
    public void testKNNExactQuery_ByteL1() {
        createTestIndex_Byte(false);
        for (byte i = 0; i < SIZE; i++) {
            addKnnDoc(INDEX_NAME, String.valueOf(i), FIELD_NAME, new float[] { i, i, i });
        }

        KNNExactQueryBuilder exactQueryBuilder = KNNExactQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(QUERY_VECTOR)
            .spaceType("l1")
            .build();

        Response searchResponse = validateKNNExactSearch(INDEX_NAME, exactQueryBuilder);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(searchResponse.getEntity()), FIELD_NAME);

        float[] expectedResults = new float[SIZE];
        for (int i = 0; i < SIZE; i++) {
            expectedResults[i] = 1 / (1 + KNNScoringUtil.l1Norm(BYTE_QUERY_VECTOR, new byte[] { (byte) i, (byte) i, (byte) i }));
        }

        assertEquals(SIZE, results.size());
        for (int i = 0; i < SIZE; i++) {
            String docId = results.get(i).getDocId();
            assertEquals(expectedResults[Integer.parseInt(docId)], results.get(i).getScore(), 0.00001);
        }
        deleteKNNIndex(INDEX_NAME);
    }

    @SneakyThrows
    public void testKNNExactQuery_Binary() {
        createTestIndex_Binary(false);
        for (byte i = 0; i < SIZE; i++) {
            addKnnDoc(INDEX_NAME, String.valueOf(i), FIELD_NAME, new float[] { i, i, i });
        }

        KNNExactQueryBuilder exactQueryBuilder = KNNExactQueryBuilder.builder().fieldName(FIELD_NAME).vector(QUERY_VECTOR).build();

        Response searchResponse = validateKNNExactSearch(INDEX_NAME, exactQueryBuilder);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(searchResponse.getEntity()), FIELD_NAME);

        float[] expectedResults = new float[SIZE];
        for (int i = 0; i < SIZE; i++) {
            expectedResults[i] = SpaceType.HAMMING.getKnnVectorSimilarityFunction()
                .compare(BYTE_QUERY_VECTOR, new byte[] { (byte) i, (byte) i, (byte) i });
        }

        assertEquals(SIZE, results.size());
        for (int i = 0; i < SIZE; i++) {
            String docId = results.get(i).getDocId();
            assertEquals(expectedResults[Integer.parseInt(docId)], results.get(i).getScore(), 0.00001);
        }
        deleteKNNIndex(INDEX_NAME);
    }

    @SneakyThrows
    public void testKNNExactQuery_LuceneBinary() {
        createTestIndex_Binary(true);
        for (byte i = 0; i < SIZE; i++) {
            addKnnDoc(INDEX_NAME, String.valueOf(i), FIELD_NAME, new float[] { i, i, i });
        }
        refreshIndex(INDEX_NAME);
        forceMergeKnnIndex(INDEX_NAME);

        KNNExactQueryBuilder exactQueryBuilder = KNNExactQueryBuilder.builder().fieldName(FIELD_NAME).vector(QUERY_VECTOR).build();

        Response searchResponse = validateKNNExactSearch(INDEX_NAME, exactQueryBuilder);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(searchResponse.getEntity()), FIELD_NAME);

        float[] expectedResults = new float[SIZE];
        for (int i = 0; i < SIZE; i++) {
            expectedResults[i] = SpaceType.HAMMING.getKnnVectorSimilarityFunction()
                .compare(BYTE_QUERY_VECTOR, new byte[] { (byte) i, (byte) i, (byte) i });
        }

        assertEquals(SIZE, results.size());
        for (int i = 0; i < SIZE; i++) {
            String docId = results.get(i).getDocId();
            assertEquals(expectedResults[Integer.parseInt(docId)], results.get(i).getScore(), 0.00001);
        }
        deleteKNNIndex(INDEX_NAME);
    }

    @SneakyThrows
    public void testKNNExactQuery_LuceneByteLINF() {
        createTestIndex_Byte(true);
        for (byte i = 0; i < SIZE; i++) {
            addKnnDoc(INDEX_NAME, String.valueOf(i), FIELD_NAME, new float[] { i, i, i });
        }
        refreshIndex(INDEX_NAME);
        forceMergeKnnIndex(INDEX_NAME);

        KNNExactQueryBuilder exactQueryBuilder = KNNExactQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(QUERY_VECTOR)
            .spaceType("linf")
            .build();

        Response searchResponse = validateKNNExactSearch(INDEX_NAME, exactQueryBuilder);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(searchResponse.getEntity()), FIELD_NAME);

        float[] expectedResults = new float[SIZE];
        for (int i = 0; i < SIZE; i++) {
            expectedResults[i] = 1 / (1 + KNNScoringUtil.lInfNorm(BYTE_QUERY_VECTOR, new byte[] { (byte) i, (byte) i, (byte) i }));
        }

        assertEquals(SIZE, results.size());
        for (int i = 0; i < SIZE; i++) {
            String docId = results.get(i).getDocId();
            assertEquals(expectedResults[Integer.parseInt(docId)], results.get(i).getScore(), 0.00001);
        }
        deleteKNNIndex(INDEX_NAME);
    }

    @SneakyThrows
    public void testKNNExactQuery_WithFiltering() {
        createTestIndex(false);
        for (byte i = 0; i < SIZE; i++) {
            String color = i % 2 == 0 ? "blue" : "red";
            addKnnDocWithAttributes(String.valueOf(i), new float[] { i, i, i }, ImmutableMap.of(FIELD_NAME_FILTER, color));
        }
        refreshIndex(INDEX_NAME);
        forceMergeKnnIndex(INDEX_NAME);

        XContentBuilder queryBuilder = XContentFactory.jsonBuilder().startObject().startObject(QUERY);
        queryBuilder.startObject("bool");
        queryBuilder.startArray("filter");
        queryBuilder.startObject().startObject("term").startObject(FIELD_NAME_FILTER);
        queryBuilder.field("value", "blue");
        queryBuilder.endObject().endObject().endObject();
        queryBuilder.endArray();
        queryBuilder.startArray("must");
        queryBuilder.startObject().startObject(EXACT_KNN).startObject(FIELD_NAME);
        queryBuilder.field(VECTOR, QUERY_VECTOR);
        queryBuilder.field("space_type", "innerproduct");
        queryBuilder.endObject().endObject().endObject();
        queryBuilder.endArray();
        queryBuilder.endObject().endObject().endObject();

        Response searchResponse = searchKNNIndex(INDEX_NAME, queryBuilder, SIZE);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(searchResponse.getEntity()), FIELD_NAME);

        float[] expectedResults = new float[(SIZE / 2) + 1];
        for (int i = 0; i < SIZE; i += 2) {
            expectedResults[i / 2] = SpaceType.INNER_PRODUCT.getKnnVectorSimilarityFunction()
                .compare(QUERY_VECTOR, new float[] { i, i, i });
        }

        assertEquals((SIZE / 2) + 1, results.size());
        for (int i = 0; i < results.size(); i++) {
            String docId = results.get(i).getDocId();
            assertEquals(expectedResults[Integer.parseInt(docId) / 2], results.get(i).getScore(), 0.00001);
        }
        deleteKNNIndex(INDEX_NAME);
    }

    public void createTestIndex(boolean isLucene) throws IOException {
        String mapping;
        if (isLucene) {
            mapping = KNNJsonIndexMappingsBuilder.builder()
                .fieldName(FIELD_NAME)
                .dimension(DIMENSION)
                .method(getLuceneMethod())
                .build()
                .getIndexMapping();
        } else {
            mapping = KNNJsonIndexMappingsBuilder.builder().fieldName(FIELD_NAME).dimension(DIMENSION).build().getIndexMapping();
        }
        createKnnIndex(INDEX_NAME, mapping);
    }

    public void createTestIndex_Byte(boolean isLucene) throws IOException {
        String mapping;
        if (isLucene) {
            mapping = KNNJsonIndexMappingsBuilder.builder()
                .fieldName(FIELD_NAME)
                .dimension(DIMENSION)
                .vectorDataType(VectorDataType.BYTE.getValue())
                .method(getLuceneMethod())
                .build()
                .getIndexMapping();
        } else {
            mapping = KNNJsonIndexMappingsBuilder.builder()
                .fieldName(FIELD_NAME)
                .dimension(DIMENSION)
                .vectorDataType(VectorDataType.BYTE.getValue())
                .build()
                .getIndexMapping();
        }
        createKnnIndex(INDEX_NAME, mapping);
    }

    public void createTestIndex_Binary(boolean isLucene) throws IOException {
        String mapping = KNNJsonIndexMappingsBuilder.builder()
            .fieldName(FIELD_NAME)
            .dimension(DIMENSION * 8)
            .vectorDataType(VectorDataType.BINARY.getValue())
            .method(getBinaryMethod(isLucene))
            .build()
            .getIndexMapping();
        createKnnIndex(INDEX_NAME, mapping);
    }

    private Response validateKNNExactSearch(String testIndex, KNNExactQueryBuilder knnExactQueryBuilder) throws Exception {
        XContentBuilder builder = XContentFactory.jsonBuilder().startObject().startObject("query");
        knnExactQueryBuilder.doXContent(builder, ToXContent.EMPTY_PARAMS);
        builder.endObject().endObject();
        return searchKNNIndex(testIndex, builder, SIZE);
    }

    private KNNJsonIndexMappingsBuilder.Method getLuceneMethod() {
        return KNNJsonIndexMappingsBuilder.Method.builder().methodName(METHOD_HNSW).engine(KNNEngine.LUCENE.getName()).build();
    }

    private KNNJsonIndexMappingsBuilder.Method getBinaryMethod(boolean isLucene) {
        return KNNJsonIndexMappingsBuilder.Method.builder()
            .methodName(METHOD_HNSW)
            .spaceType(SpaceType.HAMMING.getValue())
            .engine(isLucene ? KNNEngine.LUCENE.getName() : KNNEngine.FAISS.getName())
            .build();
    }
}
