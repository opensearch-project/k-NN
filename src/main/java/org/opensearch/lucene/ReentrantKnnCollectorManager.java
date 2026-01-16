/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.lucene;

import lombok.RequiredArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.lucene90.IndexedDISI;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.search.knn.KnnSearchStrategy;

import java.io.IOException;
import java.util.Map;

/**
 * A {@link KnnCollectorManager} that enables re-entrant (multi-phase) KNN vector search
 * by seeding the HNSW graph search with document IDs collected from a prior search phase.
 * <p>
 * This implementation reuses top-ranked results (from the 1st-phase search presumably)
 * as entry points for a 2nd-phase vector search. It converts previously collected
 * {@link TopDocs} into corresponding vector entry points using
 * {@link SeededTopDocsDISI} and {@link SeededMappedDISI}, enabling the internal searcher to start from these known points
 * instead of beginning from random or default graph entry nodes.
 * <p>
 * See <a href="https://github.com/apache/lucene/blob/71e822e6240878018a6ff3c28381a0d88bebdc72/lucene/core/src/java/org/apache/lucene/search/AbstractKnnVectorQuery.java#L368">...</a>
 */
@Log4j2
@RequiredArgsConstructor
public class ReentrantKnnCollectorManager implements KnnCollectorManager {

    // The underlying (delegate) KNN collector manager used to create collectors.
    private final KnnCollectorManager knnCollectorManager;

    // Mapping from segment ordinal to previously collected {@link TopDocs}.
    private final Map<Integer, TopDocs> segmentOrdToResults;

    // Query vector used for scoring during vector similarity search.
    private final float[] query;

    // Name of the vector field being searched.
    private final String field;

    /**
     * Creates a new {@link KnnCollector} for the given segment.
     * <p>
     * If 1st-phase results are available for the segment, this collector
     * will seed the vector search with those document IDs. The document IDs
     * are mapped to vector indices using {@link SeededMappedDISI}, which enables
     * the HNSW search to begin from those known entry points.
     * <p>
     * If no prior results exist or no vector scorer is available, the method
     * falls back to a delegate collector.
     *
     * @param visitLimit the maximum number of graph nodes that can be visited
     * @param searchStrategy the search strategy to use (e.g., HNSW or brute-force)
     * @param ctx the leaf reader context for the current segment
     * @return a seeded {@link KnnCollector} that reuses prior phase entry points,
     *         or a delegate collector if no seeding is possible
     * @throws IOException if an I/O error occurs during setup
     */
    @Override
    public KnnCollector newCollector(int visitLimit, KnnSearchStrategy searchStrategy, LeafReaderContext ctx) throws IOException {
        // Get delegate collector for fallback or empty cases
        final KnnCollector delegateCollector = knnCollectorManager.newCollector(visitLimit, searchStrategy, ctx);
        final TopDocs seedTopDocs = segmentOrdToResults.get(ctx.ord);

        if (seedTopDocs == null || seedTopDocs.totalHits.value() == 0) {
            log.warn("Seed top docs was empty, expected non-empty top results to be given.");
            // Normally shouldn't happen — indicates missing or empty seed results
            assert false;
            return delegateCollector;
        }

        // Obtain the per-segment vector values
        final LeafReader reader = ctx.reader();
        final FloatVectorValues vectorValues = reader.getFloatVectorValues(field);
        if (vectorValues == null) {
            log.error("Acquired null {} for field [{}]", FloatVectorValues.class.getSimpleName(), field);
            // Validates the field exists, otherwise throws informative exception
            FloatVectorValues.checkField(reader, field);
            return null;
        }

        // Create a vector scorer for the query vector
        final VectorScorer scorer = vectorValues.scorer(query);

        if (scorer == null) {
            log.error("Acquired null {} for field [{}]", VectorScorer.class.getSimpleName(), field);
            // Normally shouldn't happen
            assert false;
            return delegateCollector;
        }

        // Get DocIdSetIterator from scorer
        DocIdSetIterator vectorIterator = scorer.iterator();

        // Convert to an indexed iterator if possible (for sparse vectors)
        // Note that we're extracting DISI from Lucene's flat vector.
        if (vectorIterator instanceof IndexedDISI indexedDISI) {
            vectorIterator = IndexedDISI.asDocIndexIterator(indexedDISI);
        }

        // Map seed document IDs to vector indices to use as HNSW entry points
        if (vectorIterator instanceof KnnVectorValues.DocIndexIterator indexIterator) {
            DocIdSetIterator seedDocs = new SeededMappedDISI(indexIterator, new SeededTopDocsDISI(seedTopDocs));
            return knnCollectorManager.newCollector(
                visitLimit,
                new KnnSearchStrategy.Seeded(seedDocs, seedTopDocs.scoreDocs.length, searchStrategy),
                ctx
            );
        }

        log.error(
            "`vectorIterator` was not one of [{}, {}] and was {}",
            IndexedDISI.class.getSimpleName(),
            KnnVectorValues.DocIndexIterator.class.getSimpleName(),
            vectorIterator == null ? "null" : vectorIterator.getClass().getSimpleName()
        );

        // This should not occur; fallback to delegate to prevent infinite loops
        assert false;
        return delegateCollector;
    }
}
