/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.warmup;

import lombok.Setter;
import lombok.experimental.Accessors;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.FilterDirectory;
import org.opensearch.common.lucene.Lucene;

/**
 * Factory for field warm up strategy
 */
@Accessors(chain = true)
public class FieldWarmUpStrategyFactory {
    @Setter
    private Directory directory;
    @Setter
    private LeafReader leafReader;

    /**
     * Build field warm up strategy
     *
     * @return field warm up strategy
     */
    public FieldWarmUpStrategy build() {
        final Directory bottomDirectory = FilterDirectory.unwrap(directory);
        if (bottomDirectory instanceof FSDirectory) {
            return new FullFieldWarmUpStrategy(Lucene.segmentReader(leafReader), bottomDirectory);
        } else {
            return new PartialFieldWarmUpStrategy(Lucene.segmentReader(leafReader));
        }
    }
}
