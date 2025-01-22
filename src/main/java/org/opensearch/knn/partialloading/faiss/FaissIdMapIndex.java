/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.partialloading.faiss;

import lombok.AllArgsConstructor;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.partialloading.search.DocIdGrouper;
import org.opensearch.knn.partialloading.search.IdAndDistance;
import org.opensearch.knn.partialloading.search.MatchDocSelector;
import org.opensearch.knn.partialloading.search.PartialLoadingSearchParameters;

import java.io.IOException;

/**
 * A FAISS index with an ID mapping that maps the internal vector ID to a logical ID, along with the actual vector index.
 * It first delegates the vector search to its nested vector index, then transforms the vector ID into a logical index that is
 * understandable by upstream components. This is particularly useful when not all Lucene documents are indexed with a vector field.
 * For example, if 70% of the documents have a vector field and the remaining 30% do not, the FAISS vector index will still assign
 * increasing and continuous vector IDs starting from 0.
 * However, these IDs only cover the sparse 30% of Lucene documents, so an ID mapping is needed to convert the internal physical vector ID
 * into the corresponding Lucene document ID.
 * If the mapping is an identity mapping, where each `i` is mapped to itself, we omit storing it to save memory.
 */
public class FaissIdMapIndex extends FaissIndex {
    public static final String IXMP = "IxMp";

    private FaissIndex nestedIndex;
    private long[] vectorIdToDocIdMapping;

    /**
     * Partially load id mapping and its nested index to which vector searching will be delegated.
     *
     * @param input An input stream for a FAISS HNSW graph file, allowing access to the neighbor list and vector locations.
     * @return {@link FaissIdMapIndex} instance consists of index hierarchy.
     * @throws IOException
     */
    public static FaissIdMapIndex partiallyLoad(IndexInput input) throws IOException {
        FaissIdMapIndex faissIdMapIndex = new FaissIdMapIndex();
        readCommonHeader(input, faissIdMapIndex);
        faissIdMapIndex.nestedIndex = FaissIndex.partiallyLoad(input);

        // Load `idMap`
        final long numElements = input.readLong();
        long[] vectorIdToDocIdMapping = new long[(int) numElements];
        input.readLongs(vectorIdToDocIdMapping, 0, vectorIdToDocIdMapping.length);

        // If `idMap` is an identity function that maps `i` to `i`, then we don't need to keep it.
        for (int i = 0; i < vectorIdToDocIdMapping.length; i++) {
            if (vectorIdToDocIdMapping[i] != i) {
                // Only keep it if it's not an identify mapping.
                faissIdMapIndex.vectorIdToDocIdMapping = vectorIdToDocIdMapping;
                break;
            }
        }

        return faissIdMapIndex;
    }

    /**
     * Delegates the vector search to its nested index, then remaps the internal vector IDs to logical IDs.
     * The remapping process considers two parameters provided in `searchParameters`.
     * <p>
     * - The first parameter is the **selector**, which accepts a logical document ID to determine whether the document should be
     * included in the results.
     * - The second parameter is the **grouper**, which also accepts a logical document ID and returns a group ID that represents or
     * replaces the document ID. A typical use case for this is the case where we indexed a document having sub-vectors. In this case,
     * all sub-vectors share the same `group ID` pointing to a document containing them.
     * <p>
     * For more details on ranking, refer to {@link org.opensearch.knn.partialloading.search.GroupedDistanceMaxHeap}.
     * <p>
     * If non-null parameters are provided, this wraps them to translate the physical vector ID into a logical ID. This allows the
     * selector and grouper to correctly identify the document and ensure the proper procedure is followed.
     * Ex: vector id -> remaps -> logical document id -> selector or grouper
     *
     * @param indexInput An input stream for a FAISS HNSW graph file, allowing access to the neighbor list and vector locations.
     * @param results A result array containing non-null pairs of vector IDs and their distances. After the search, it is updated by
     *                extracting elements from the result max-heap.
     * @param searchParameters HNSW search parameters, including efSearch, allow customization. If efSearch is provided, it will override
     *                        the default value.
     * @throws IOException
     */
    @Override
    public void search(IndexInput indexInput, IdAndDistance[] results, PartialLoadingSearchParameters searchParameters) throws IOException {
        final MatchDocSelector selector = searchParameters.getMatchDocSelector();
        final DocIdGrouper docIdGrouper = searchParameters.getDocIdGrouper();

        try {
            if (vectorIdToDocIdMapping != null && selector != null) {
                // We only have non-null idMap when it's not an identity mapping.
                // 1 = True, 0 = False
                // || identity | sel | selector wrapping |
                // ||    0     |  0  |      No need      |
                // ||    0     |  1  |      Need         |
                // ||    1     |  0  |      No need      |
                // ||    1     |  1  |      No need      |
                searchParameters.setMatchDocSelector(new DocIdSelectorTranslated(selector));
            }

            if (vectorIdToDocIdMapping != null && docIdGrouper != null) {
                // We only have non-null idMap when it's not an identity mapping.
                // 1 = True, 0 = False
                // | identity | grouper | grouper wrapping |
                // ||    0    |     0   |      No need     |
                // ||    0    |     1   |      Need        |
                // ||    1    |     0   |      No need     |
                // ||    1    |     1   |      No need     |
                searchParameters.setDocIdGrouper(new DocIdGrouperTranslated(docIdGrouper));
            }

            nestedIndex.search(indexInput, results, searchParameters);
            transformResultDocIds(results);
        } finally {
            // Recover the original selector and grouper.
            searchParameters.setMatchDocSelector(selector);
            searchParameters.setDocIdGrouper(docIdGrouper);
        }
    }

    @Override
    public String getIndexType() {
        return IXMP;
    }

    private void transformResultDocIds(IdAndDistance[] results) {
        if (vectorIdToDocIdMapping != null) {
            // Transform vector id to document id.
            // Ex: vec_id=77 -> doc_id = idMap[vec_id] = idMap[77]
            for (final IdAndDistance result : results) {
                if (result.id != IdAndDistance.INVALID_DOC_ID) {
                    result.id = (int) (vectorIdToDocIdMapping[result.id]);
                } else {
                    // Since `results` is sorted, it is guaranteed that since this point, there will no valid docs present.
                    // Ex: [doc1, doc2, doc3, -1, -1, -1, -1, -1]
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
            return nestedDocSelector.test((int) vectorIdToDocIdMapping[docId]);
        }
    }

    @AllArgsConstructor
    private class DocIdGrouperTranslated implements DocIdGrouper {
        final DocIdGrouper nestedDocIdGrouper;

        @Override
        public int getGroupId(int childDocId) {
            return nestedDocIdGrouper.getGroupId((int) vectorIdToDocIdMapping[childDocId]);
        }
    }
}
