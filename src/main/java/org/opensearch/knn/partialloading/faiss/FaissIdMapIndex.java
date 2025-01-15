/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.partialloading.faiss;

import lombok.AllArgsConstructor;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.partialloading.search.DocIdAndDistance;
import org.opensearch.knn.partialloading.search.DocIdGrouper;
import org.opensearch.knn.partialloading.search.MatchDocSelector;
import org.opensearch.knn.partialloading.search.PartialLoadingSearchParameters;

import java.io.IOException;
import java.util.Arrays;

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
        for (int i = 0; i < idMap.length; i++) {
            if (idMap[i] != i) {
                // Only keep it if it's not an identify mapping.
                faissIdMapIndex.idMap = idMap;
                break;
            }
        }

        // TMP
        if (faissIdMapIndex.idMap != null) {
            System.out.println(" ++++++++++ idMap=" + Arrays.toString(faissIdMapIndex.idMap));
        }
        // TMP

        return faissIdMapIndex;
    }

    @Override
    public void searchLeaf(IndexInput indexInput, DocIdAndDistance[] results, PartialLoadingSearchParameters searchParameters)
        throws IOException {
        final MatchDocSelector selector = searchParameters.getMatchDocSelector();
        final DocIdGrouper docIdGrouper = searchParameters.getDocIdGrouper();

        try {
            if (idMap != null && selector != null) {
                // We only have non-null idMap when it's not an identity mapping.
                // 1 = True, 0 = False
                // || identity | sel | selector wrapping |
                // ||    0     |  0  |      No need      |
                // ||    0     |  1  |      Need         |
                // ||    1     |  0  |      No need      |
                // ||    1     |  1  |      No need      |
                searchParameters.setMatchDocSelector(new DocIdSelectorTranslated(selector));
            }

            if (idMap != null && docIdGrouper != null) {
                // We only have non-null idMap when it's not an identity mapping.
                // 1 = True, 0 = False
                // | identity | grouper | grouper wrapping |
                // ||    0    |     0   |      No need     |
                // ||    0    |     1   |      Need        |
                // ||    1    |     0   |      No need     |
                // ||    1    |     1   |      No need     |
                searchParameters.setDocIdGrouper(new DocIdGrouperTranslated(docIdGrouper));
            }

            nestedIndex.searchLeaf(indexInput, results, searchParameters);
            transformResultDocIds(results);
        } finally {
            searchParameters.setMatchDocSelector(selector);
            searchParameters.setDocIdGrouper(docIdGrouper);
        }
    }

    @Override
    public String getIndexType() {
        return IXMP;
    }

    private void transformResultDocIds(DocIdAndDistance[] results) {
        if (idMap != null) {
            // Transform doc id.
            // Ex: doc_id=-77 -> doc_id = idMap[-doc_id] = idMap[77]
            for (final DocIdAndDistance result : results) {
                if (result.id != DocIdAndDistance.INVALID_DOC_ID) {
                    result.id = (int) (idMap[result.id]);
                } else {
                    // Since `results` is sorted, it is guaranteed that since this point, no valid docs present.
                    break;
                }
            }
        }
    }

    @AllArgsConstructor
    private class DocIdSelectorTranslated implements MatchDocSelector {
        final MatchDocSelector nestedDocSelector;

        @Override
        public boolean test(int docId) {
            return nestedDocSelector.test((int) idMap[docId]);
        }
    }

    @AllArgsConstructor
    private class DocIdGrouperTranslated implements DocIdGrouper {
        final DocIdGrouper nestedDocIdGrouper;

        @Override
        public int getGroupId(int childDocId) {
            return nestedDocIdGrouper.getGroupId((int) idMap[childDocId]);
        }
    }
}
