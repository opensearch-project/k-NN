/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import org.apache.lucene.search.TopKnnCollector;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.KNNEngine;

/**
 * A specialized collector for approximate k-nearest neighbor search results.
 * This collector extends Lucene's TopKnnCollector to handle approximate k-NN search
 * with support for different space types and scoring mechanisms through the KNNEngine.
 */
public class TopApproxKnnCollector extends TopKnnCollector {

    private final KNNEngine engine;
    private final SpaceType spaceType;

    public TopApproxKnnCollector(int k, KNNEngine engine, SpaceType spaceType) {
        super(k, Integer.MAX_VALUE);
        this.spaceType = spaceType;
        this.engine = engine;
    }

    /**
     * Collects a document with its similarity score, converting the raw similarity
     * to a final score using the configured KNN engine and space type.
     * @param docId of the vector to collect
     * @param similarity its calculated similarity
     * @return true if the document was competitive (i.e., collected), false otherwise
     */
    @Override
    public boolean collect(int docId, float similarity) {
        return super.collect(docId, engine.score(similarity, spaceType));
    }
}
