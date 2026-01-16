/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.search.processor.mmr;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.search.fetch.subphase.FetchSourceContext;

import java.util.Map;

/**
 * A DTO to hold the context for MMR rerank
 */
@Data
@NoArgsConstructor
public class MMRRerankContext {
    private Integer originalQuerySize;
    private Float diversity;
    private FetchSourceContext originalFetchSourceContext;
    private SpaceType spaceType;
    // The default path if we cannot find the path based on the index
    private String vectorFieldPath;
    private VectorDataType vectorDataType;
    // To support the case that we have different vector field paths in different indices
    private Map<String, String> indexToVectorFieldPathMap;
}
