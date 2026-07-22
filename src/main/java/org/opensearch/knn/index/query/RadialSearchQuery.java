/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.MatchNoDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.search.knn.KnnSearchStrategy;
import org.apache.lucene.search.knn.TopKnnCollectorManager;
import org.apache.lucene.util.Bits;
import org.opensearch.knn.index.query.common.QueryUtils;
import org.opensearch.knn.index.query.memoryoptsearch.RadiusVectorSimilarityCollector;
import org.opensearch.lucene.ReentrantKnnCollectorManager;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.Callable;

/**
 * Two-phase radial (similarity-threshold) search query.
 * <p>
 * Phase 1 runs a top-k ANN search (k = ef_search) to discover high-quality seed entry points
 * in the HNSW graph. Phase 2 runs a radial search seeded from those entry points, collecting
 * all vectors at or above the similarity threshold.
 * <p>
 * This query is engine-agnostic: it calls {@code LeafReader.searchNearestVectors()} which works
 * for both Lucene-native HNSW indices and Faiss memory-optimized indices (via their respective
 * {@code KnnVectorsReader} implementations).
 */
@Log4j2
public class RadialSearchQuery extends Query {
    private static final KnnSearchStrategy.Hnsw DEFAULT_HNSW_SEARCH_STRATEGY = new KnnSearchStrategy.Hnsw(0);

    private final String field;
    private final float[] target;
    private final byte[] byteTarget;
    private final float similarity;
    private final int efSearch;
    private final Query filter;

    public RadialSearchQuery(String field, float[] target, float similarity, int efSearch, Query filter) {
        this.field = Objects.requireNonNull(field);
        this.target = Objects.requireNonNull(target);
        this.byteTarget = null;
        this.similarity = similarity;
        this.efSearch = efSearch;
        this.filter = filter;
    }

    public RadialSearchQuery(String field, byte[] byteTarget, float similarity, int efSearch, Query filter) {
        this.field = Objects.requireNonNull(field);
        this.target = null;
        this.byteTarget = Objects.requireNonNull(byteTarget);
        this.similarity = similarity;
        this.efSearch = efSearch;
        this.filter = filter;
    }

    @Override
    public Weight createWeight(IndexSearcher searcher, ScoreMode scoreMode, float boost) throws IOException {
        final Weight filterWeight;
        if (filter != null) {
            Query rewritten = searcher.rewrite(filter);
            filterWeight = searcher.createWeight(rewritten, ScoreMode.COMPLETE_NO_SCORES, 1f);
        } else {
            filterWeight = null;
        }

        List<LeafReaderContext> leaves = searcher.getIndexReader().leaves();
        List<Callable<TopDocs>> tasks = new ArrayList<>(leaves.size());
        for (LeafReaderContext ctx : leaves) {
            tasks.add(() -> searchLeaf(ctx, filterWeight, searcher));
        }

        List<TopDocs> leafResults = searcher.getTaskExecutor().invokeAll(tasks);

        // Merge all per-leaf results; add docBase offsets
        int totalHits = 0;
        for (int i = 0; i < leafResults.size(); i++) {
            TopDocs leafTopDocs = leafResults.get(i);
            int docBase = leaves.get(i).docBase;
            for (ScoreDoc sd : leafTopDocs.scoreDocs) {
                sd.doc += docBase;
            }
            totalHits += leafTopDocs.scoreDocs.length;
        }

        // Flatten all results into a single TopDocs
        ScoreDoc[] allDocs = new ScoreDoc[totalHits];
        int idx = 0;
        for (TopDocs leafTopDocs : leafResults) {
            for (ScoreDoc sd : leafTopDocs.scoreDocs) {
                allDocs[idx++] = sd;
            }
        }
        TopDocs merged = new TopDocs(new TotalHits(totalHits, TotalHits.Relation.EQUAL_TO), allDocs);

        if (merged.scoreDocs.length == 0) {
            return new MatchNoDocsQuery().createWeight(searcher, scoreMode, boost);
        }

        return QueryUtils.getInstance().createDocAndScoreQuery(searcher.getIndexReader(), merged).createWeight(searcher, scoreMode, boost);
    }

    private TopDocs searchLeaf(LeafReaderContext ctx, Weight filterWeight, IndexSearcher searcher) throws IOException {
        final LeafReader reader = ctx.reader();
        final boolean isByteVector = byteTarget != null;

        if (isByteVector) {
            if (reader.getByteVectorValues(field) == null) {
                return new TopDocs(new TotalHits(0, TotalHits.Relation.EQUAL_TO), new ScoreDoc[0]);
            }
        } else {
            if (reader.getFloatVectorValues(field) == null) {
                return new TopDocs(new TotalHits(0, TotalHits.Relation.EQUAL_TO), new ScoreDoc[0]);
            }
        }

        // Determine accepted docs and visit limit
        final AcceptDocs acceptDocs;
        final int visitLimit;
        if (filterWeight != null) {
            Scorer scorer = filterWeight.scorer(ctx);
            if (scorer == null) {
                return new TopDocs(new TotalHits(0, TotalHits.Relation.EQUAL_TO), new ScoreDoc[0]);
            }
            Bits liveDocs = reader.getLiveDocs();
            acceptDocs = AcceptDocs.fromIteratorSupplier(() -> scorer.iterator(), liveDocs, reader.maxDoc());
            visitLimit = acceptDocs.cost();
            if (visitLimit == 0) {
                return new TopDocs(new TotalHits(0, TotalHits.Relation.EQUAL_TO), new ScoreDoc[0]);
            }
        } else {
            acceptDocs = AcceptDocs.fromLiveDocs(reader.getLiveDocs(), reader.maxDoc());
            visitLimit = Integer.MAX_VALUE;
        }

        // Phase 1: top-k ANN with k = ef_search to find seed entry points.
        log.info("[RADIAL-DEBUG] RadialSearchQuery.searchLeaf: field={}, similarity={}, efSearch={}, maxDoc={}",
            field, similarity, efSearch, reader.maxDoc());
        KnnCollector seedCollector = new TopKnnCollectorManager(efSearch, searcher).newCollector(
            visitLimit,
            DEFAULT_HNSW_SEARCH_STRATEGY,
            ctx
        );
        if (isByteVector) {
            reader.searchNearestVectors(field, byteTarget, seedCollector, acceptDocs);
        } else {
            reader.searchNearestVectors(field, target, seedCollector, acceptDocs);
        }
        TopDocs seedTopDocs = seedCollector.topDocs();
        log.info("[RADIAL-DEBUG] Phase 1 seeds: count={}, minScore={}, maxScore={}",
            seedTopDocs.scoreDocs.length,
            seedTopDocs.scoreDocs.length > 0 ? seedTopDocs.scoreDocs[seedTopDocs.scoreDocs.length - 1].score : "N/A",
            seedTopDocs.scoreDocs.length > 0 ? seedTopDocs.scoreDocs[0].score : "N/A");

        // Phase 2: radial search, seeded from phase-1 results.
        final KnnCollectorManager radialCollectorManager;
        KnnCollectorManager baseRadialManager = (vl, strategy, context) -> new RadiusVectorSimilarityCollector(similarity, vl, strategy);

        if (seedTopDocs != null && seedTopDocs.scoreDocs.length > 0) {
            Object queryVector = isByteVector ? byteTarget : target;
            radialCollectorManager = new ReentrantKnnCollectorManager(baseRadialManager, Map.of(ctx.ord, seedTopDocs), queryVector, field);
        } else {
            radialCollectorManager = baseRadialManager;
        }

        KnnCollector radialCollector = radialCollectorManager.newCollector(visitLimit, DEFAULT_HNSW_SEARCH_STRATEGY, ctx);
        if (isByteVector) {
            reader.searchNearestVectors(field, byteTarget, radialCollector, acceptDocs);
        } else {
            reader.searchNearestVectors(field, target, radialCollector, acceptDocs);
        }
        TopDocs results = radialCollector.topDocs();

        log.info("[RADIAL-DEBUG] Phase 2 results: count={}, visited={}",
            results == null ? 0 : results.scoreDocs.length,
            radialCollector.visitedCount());

        if (results == null || results.scoreDocs.length == 0) {
            return new TopDocs(new TotalHits(0, TotalHits.Relation.EQUAL_TO), new ScoreDoc[0]);
        }

        return results;
    }

    @Override
    public String toString(String field) {
        return "RadialSearchQuery[field=" + this.field + " similarity=" + similarity + " efSearch=" + efSearch + "]";
    }

    @Override
    public void visit(QueryVisitor visitor) {
        if (visitor.acceptField(field)) {
            visitor.visitLeaf(this);
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        RadialSearchQuery that = (RadialSearchQuery) o;
        return Float.compare(that.similarity, similarity) == 0
            && efSearch == that.efSearch
            && field.equals(that.field)
            && Arrays.equals(target, that.target)
            && Arrays.equals(byteTarget, that.byteTarget)
            && Objects.equals(filter, that.filter);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(field, similarity, efSearch, filter);
        result = 31 * result + Arrays.hashCode(target);
        result = 31 * result + Arrays.hashCode(byteTarget);
        return result;
    }
}
