/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.grpc.proto.request.search.query;

import org.opensearch.index.query.QueryBuilder;
import org.opensearch.transport.grpc.proto.request.search.query.QueryBuilderProtoConverter;
import org.opensearch.protobufs.QueryContainer;

/**
 * Converter for KNN queries.
 * This class implements the QueryBuilderProtoConverter interface to provide KNN query support
 * for the gRPC transport plugin.
 */
public class KNNQueryBuilderProtoConverter implements QueryBuilderProtoConverter {

    @Override
    public QueryContainer.QueryContainerCase getHandledQueryCase() {
        return QueryContainer.QueryContainerCase.KNN;
    }

    @Override
    public QueryBuilder fromProto(QueryContainer queryContainer) {
        if (queryContainer == null || queryContainer.getQueryContainerCase() != QueryContainer.QueryContainerCase.KNN) {
            throw new IllegalArgumentException("QueryContainer does not contain a KNN query");
        }

        return KNNQueryBuilderProtoUtils.fromProto(queryContainer.getKnn());
    }
}
