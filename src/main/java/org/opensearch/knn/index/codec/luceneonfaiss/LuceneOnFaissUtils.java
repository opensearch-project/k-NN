/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.luceneonfaiss;

import org.opensearch.knn.common.KNNConstants;

import java.util.Map;

public final class LuceneOnFaissUtils {
    private LuceneOnFaissUtils() {
    }

    public static boolean isUseLuceneOnFaiss(Object mapObject) {
        if (mapObject instanceof Map) {
            Map map = (Map) mapObject;
            Object value = map.get(KNNConstants.USE_LUCENE_HNSW_SEARCHER);
            return (value instanceof Boolean) ? (Boolean) value : false;
        }

        return false;
    }
}
