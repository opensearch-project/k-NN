/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.partialloading.faiss;

import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.index.query.KNNQueryResult;
import org.opensearch.knn.partialloading.search.PartialLoadingSearchParameters;
import org.opensearch.knn.partialloading.search.ResultsCollector;

import java.io.IOException;

public class FaissIdMapIndex extends FaissIndex {
    public static final String IXMP = "IxMp";

    private FaissIndex nestedIndex;
    private long[] idMap;

    public static FaissIndex load(IndexInput input) throws IOException {
        FaissIdMapIndex faissIdMapIndex = new FaissIdMapIndex();
        readCommonHeader(input, faissIdMapIndex);
        faissIdMapIndex.nestedIndex = FaissIndex.load(input);

        // Load `idMap`
        final long numElements = input.readLong();
        long[] idMap = new long[(int) numElements];
        input.readLongs(idMap, 0, idMap.length);

        // If `idMap` is an identity function that maps `i` to `i`, then we don't need to keep it.
        boolean identityMap = true;
        for (int i = 0; i < idMap.length; i++) {
            if (idMap[i] != i) {
                identityMap = false;
                break;
            }
        }
        if (!identityMap) {
            // Only keep it if it's not an identify mapping.
            faissIdMapIndex.idMap = idMap;
        }

        return faissIdMapIndex;
    }

    @Override
    public void searchLeaf(IndexInput indexInput, ResultsCollector resultsCollector, PartialLoadingSearchParameters searchParameters)
        throws IOException {
        // TODO : params->sel
        // TODO : params->grp

        nestedIndex.searchLeaf(indexInput, resultsCollector, searchParameters);

        transformResultDocIds(resultsCollector);
    }

    @Override
    public String getIndexType() {
        return IXMP;
    }

    private void transformResultDocIds(ResultsCollector resultsCollector) {
        KNNQueryResult[] results = resultsCollector.getResults();
        if (idMap == null) {
            // Identify mapping, transform `i` (which is a negative) to `-i`.
            // Ex: doc_id=-77 -> doc_id=77
            for (final KNNQueryResult result : results) {
                if (result.getId() < 0) {
                    result.reset(-result.getId(), result.getScore());
                }
            }
        } else {
            // Transform doc id.
            // Ex: doc_id=-77 -> doc_id= idMap[-doc_id] = idMap[77]
            for (final KNNQueryResult result : results) {
                if (result.getId() < 0) {
                    result.reset((int) idMap[-result.getId()], result.getScore());
                }
            }
        }
    }
}
