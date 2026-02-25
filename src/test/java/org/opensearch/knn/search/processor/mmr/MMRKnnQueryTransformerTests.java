/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.search.processor.mmr;

import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.cluster.metadata.MappingMetadata;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.index.Index;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.transport.client.Client;

import java.util.List;
import java.util.Map;

import static org.mockito.Mockito.*;
import static org.opensearch.knn.common.KNNConstants.TOP_LEVEL_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.TYPE;

public class MMRKnnQueryTransformerTests extends MMRTestCase {
    private Client client;
    private MMRKnnQueryTransformer transformer;
    private KNNQueryBuilder queryBuilder;
    private ActionListener<Void> listener;
    private MMRTransformContext transformContext;
    private MMRRerankContext processingContext;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        client = mock(Client.class);
        transformer = new MMRKnnQueryTransformer();
        queryBuilder = mock(KNNQueryBuilder.class);
        listener = mock(ActionListener.class);
        processingContext = new MMRRerankContext();
        transformContext = new MMRTransformContext(10, processingContext, List.of(), List.of(), null, null, null, client, false);
    }

    public void testTransform_whenNoMaxDistanceOrMinScore_thenSetsK() {
        when(queryBuilder.getMaxDistance()).thenReturn(null);
        when(queryBuilder.getMinScore()).thenReturn(null);

        transformer.transform(queryBuilder, listener, transformContext);

        verify(queryBuilder).setK(10);
    }

    public void testTransform_whenMinScore_thenNotSetsK() {
        when(queryBuilder.getMaxDistance()).thenReturn(null);
        when(queryBuilder.getMinScore()).thenReturn(0.5f); // non-null minScore

        transformer.transform(queryBuilder, listener, transformContext);

        verify(queryBuilder, never()).setK(anyInt());
    }

    public void testTransform_whenVectorFieldInfoAlreadyResolved_thenEarlyExits() {
        transformContext = new MMRTransformContext(
            10,
            processingContext,
            List.of(),
            List.of(),
            null,
            "vector.field.path",
            null,
            client,
            true
        );

        transformer.transform(queryBuilder, listener, transformContext);

        verify(listener).onResponse(null);
        verifyNoMoreInteractions(client);
    }

    public void testTransform_whenNoUserProvidedVectorFieldPath_thenResolveSpaceType() {
        String indexName = "test-index";
        String vectorFieldName = "vectorField";
        IndexMetadata indexMetadata = mock(IndexMetadata.class);
        MappingMetadata mappingMetadata = mock(MappingMetadata.class);
        Map<String, Object> mapping = Map.of(
            indexName,
            Map.of(
                "properties",
                Map.of(
                    vectorFieldName,
                    Map.of(TYPE, KNNVectorFieldMapper.CONTENT_TYPE, TOP_LEVEL_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
                )
            )
        );
        when(indexMetadata.getIndex()).thenReturn(new Index(indexName, "uuid"));
        when(indexMetadata.mapping()).thenReturn(mappingMetadata);
        when(mappingMetadata.sourceAsMap()).thenReturn(mapping);
        when(queryBuilder.fieldName()).thenReturn(vectorFieldName);

        transformContext = new MMRTransformContext(
            10,
            processingContext,
            List.of(indexMetadata),
            List.of(),
            null,
            null,
            null,
            client,
            false
        );

        transformer.transform(queryBuilder, listener, transformContext);

        verify(listener).onResponse(null);
        assertEquals(vectorFieldName, processingContext.getVectorFieldPath());
        assertEquals(SpaceType.L2, processingContext.getSpaceType());
    }
}
