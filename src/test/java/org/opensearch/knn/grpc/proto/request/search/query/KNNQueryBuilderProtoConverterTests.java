/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.grpc.proto.request.search.query;

import org.junit.Before;
import org.junit.Test;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.protobufs.KnnQuery;
import org.opensearch.protobufs.QueryContainer;
import org.opensearch.test.OpenSearchTestCase;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class KNNQueryBuilderProtoConverterTests extends OpenSearchTestCase {

    private KNNQueryBuilderProtoConverter converter;
    private QueryContainer queryContainer;
    private KnnQuery knnQuery;

    @Before
    public void setup() {
        converter = new KNNQueryBuilderProtoConverter();
        queryContainer = mock(QueryContainer.class);
        knnQuery = mock(KnnQuery.class);
    }

    @Test
    public void testGetHandledQueryCase() {
        assertEquals(QueryContainer.QueryContainerCase.KNN, converter.getHandledQueryCase());
    }

    @Test
    public void testFromProto_validQuery() {
        // Setup
        when(queryContainer.getQueryContainerCase()).thenReturn(QueryContainer.QueryContainerCase.KNN);
        when(queryContainer.getKnn()).thenReturn(knnQuery);

        // Mock the KNNQueryBuilderProtoUtils.fromProto method using PowerMock
        KNNQueryBuilder expectedQueryBuilder = mock(KNNQueryBuilder.class);

        // Test
        QueryBuilder result = converter.fromProto(queryContainer);

        // Verify
        assertNotNull(result);
    }

    @Test
    public void testFromProto_nullQueryContainer() {
        // Test
        IllegalArgumentException exception = expectThrows(IllegalArgumentException.class, () -> converter.fromProto(null));

        // Verify
        assertEquals("QueryContainer does not contain a KNN query", exception.getMessage());
    }

    @Test
    public void testFromProto_wrongQueryContainerCase() {
        // Setup
        when(queryContainer.getQueryContainerCase()).thenReturn(QueryContainer.QueryContainerCase.BOOL);

        // Test
        IllegalArgumentException exception = expectThrows(
            IllegalArgumentException.class,
            () -> converter.fromProto(queryContainer)
        );

        // Verify
        assertEquals("QueryContainer does not contain a KNN query", exception.getMessage());
    }
}
