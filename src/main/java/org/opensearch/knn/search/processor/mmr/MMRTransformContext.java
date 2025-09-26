/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.search.processor.mmr;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NonNull;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.transport.client.Client;

import java.util.List;

/**
 * A DTO to hold the context for MMR query transformer to transform the request
 */
@Data
@AllArgsConstructor
public class MMRTransformContext {
    @NonNull
    private final Integer candidates;
    // During transform, we will also collect some info for MMR rerank processor to use later
    @NonNull
    private final MMRRerankContext mmrRerankContext;
    // During transform, we need to figure out the knn_vector space type based on the index metadata
    @NonNull
    private final List<IndexMetadata> localIndexMetadataList;
    @NonNull
    private final List<String> remoteIndices;
    private final SpaceType userProvidedSpaceType;
    private final String userProvidedVectorFieldPath;
    private final VectorDataType userProvidedVectorDataType;
    private final Client client;
    private boolean isVectorFieldInfoResolved;
}
