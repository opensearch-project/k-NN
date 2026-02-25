/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.generate;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.opensearch.knn.generate.DocumentsGenerator.NUM_CHILD_DOCS;
import static org.opensearch.knn.generate.SearchTestHelper.takePortions;

/**
 * Indexing type dictates how {@link DocumentsGenerator} should generate documents.
 */
public enum IndexingType {
    DENSE {
        @Override
        public List<Integer> generateDocumentIds(int totalNumberOfDocs) {
            return IntStream.rangeClosed(0, totalNumberOfDocs - 1).boxed().collect(Collectors.toList());
        }
    },
    SPARSE {
        @Override
        public List<Integer> generateDocumentIds(int totalNumberOfDocs) {
            List<Integer> docIds = new ArrayList<>(IntStream.rangeClosed(0, totalNumberOfDocs - 1).boxed().collect(Collectors.toList()));

            // Take only 80% docs. e.g. 20% docs won't have vector field.
            return takePortions(docIds, 0.8);
        }
    },
    DENSE_NESTED {
        @Override
        public List<Integer> generateDocumentIds(int totalNumberOfDocs) {
            final int NUM_CHILD_DOCS = 5;

            final int numDocsHavingVector = NUM_CHILD_DOCS * totalNumberOfDocs;
            final List<Integer> docIds = new ArrayList<>(numDocsHavingVector);
            for (int i = 0; i < totalNumberOfDocs; i++) {
                for (int j = 0; j < NUM_CHILD_DOCS; ++j) {
                    docIds.add(i * (NUM_CHILD_DOCS + 1) + j);
                }
            }
            // Ex: [[0, 1, 2, 3, 4],
            // [6, 7, 8, 9, 10],
            // [12, ...]
            // Note that doc=5, doc=11 are parent document.
            return docIds;
        }

        @Override
        public boolean isNested() {
            return true;
        }
    },
    SPARSE_NESTED {
        @Override
        public List<Integer> generateDocumentIds(int totalNumberOfDocs) {
            // Ex: [[0, 1, 2, 3, 4],
            // [12, 13, 14, 15, 16],
            // [18, ...]
            // Note that docs in [6, 11] don't have vector fields.
            List<Integer> docIds = new ArrayList<>();
            int nextDocId = 0;
            for (int i = 0; i < totalNumberOfDocs; i++) {
                // Only take 80%, or if it's last index
                // Why force it to add child docs at the last index?
                // -> So that we can always restore parent ids deterministically.
                if (i == (totalNumberOfDocs - 1) || ThreadLocalRandom.current().nextFloat(1f) >= 0.8f) {
                    // This doc has vector
                    for (int j = 0; j < NUM_CHILD_DOCS; j++) {
                        docIds.add(nextDocId++);
                    }

                    // Visit a parent doc
                    ++nextDocId;
                } else {
                    // This doc don't have vector
                    ++nextDocId;
                }
            }

            return docIds;
        }

        @Override
        public boolean isNested() {
            return true;
        }
    };

    public abstract List<Integer> generateDocumentIds(int totalNumberOfDocs);

    public boolean isNested() {
        return false;
    }
}
