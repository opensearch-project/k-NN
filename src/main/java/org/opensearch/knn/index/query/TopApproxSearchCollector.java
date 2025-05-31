/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import org.apache.lucene.search.TopKnnCollector;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.KNNEngine;

public class TopApproxSearchCollector extends TopKnnCollector {

    private final KNNEngine engine;
    private final SpaceType spaceType;

    public TopApproxSearchCollector(int k, KNNEngine engine, SpaceType spaceType) {
        super(k, Integer.MAX_VALUE);
        this.spaceType = spaceType;
        this.engine = engine;
    }

    @Override
    public boolean collect(int docId, float similarity) {
        return super.collect(docId, engine.score(similarity, spaceType));
    }
}
