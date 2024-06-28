/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.common;

import java.util.Map;

import static org.opensearch.knn.index.util.Faiss.FAISS_BINARY_INDEX_DESCRIPTION_PREFIX;

public class KNNFaissUtil {
    public boolean isBinaryIndex(Map<String, Object> parameters) {
        return parameters.get(KNNConstants.INDEX_DESCRIPTION_PARAMETER) != null
            && parameters.get(KNNConstants.INDEX_DESCRIPTION_PARAMETER).toString().startsWith(FAISS_BINARY_INDEX_DESCRIPTION_PREFIX);
    }
}
