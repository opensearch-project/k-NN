/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.nativelib;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchNoDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.knn.TopKnnCollectorManager;
import org.apache.lucene.util.Bits;
import org.opensearch.common.StopWatch;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.query.ExactSearcher;
import org.opensearch.knn.index.query.KNNQuery;
import org.opensearch.knn.index.query.KNNWeight;
import org.opensearch.knn.index.query.TopDocsDISI;
import org.opensearch.knn.index.query.PerLeafResult;
import org.opensearch.knn.index.query.ResultUtil;
import org.opensearch.knn.index.query.common.QueryUtils;
import org.opensearch.knn.index.query.memoryoptsearch.MemoryOptimizedKNNWeight;
import org.opensearch.knn.index.query.memoryoptsearch.optimistic.OptimisticSearchStrategyUtils;
import org.opensearch.lucene.ReentrantKnnCollectorManager;
import org.opensearch.knn.index.query.rescore.RescoreContext;
import org.opensearch.knn.profile.KNNProfileUtil;
import org.opensearch.knn.profile.LongMetric;
import org.opensearch.knn.profile.query.KNNMetrics;
import org.opensearch.search.profile.AbstractProfileBreakdown;
import org.opensearch.search.profile.ContextualProfileBreakdown;
import org.opensearch.search.profile.query.QueryProfiler;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.stream.Collectors;

import static org.opensearch.knn.profile.StopWatchUtils.startStopWatch;
import static org.opensearch.knn.profile.StopWatchUtils.stopStopWatchAndLog;

/**
 * {@link KNNQuery} executes approximate nearest neighbor search (ANN) on a segment level.
 * {@link NativeEngineKnnVectorQuery} executes approximate nearest neighbor search but gives
 * us the control to combine the top k results in each leaf and post process the results just
 * for k-NN query if required. This is done by overriding rewrite method to execute ANN on each leaf
 * {@link KNNQuery} does not give the ability to post process segment results.
 */
@Log4j2
@Getter
@RequiredArgsConstructor
public class NativeEngineKnnVectorQuery extends Query {
    /**
     * A special flag used for testing purposes that forces execution of the second (exact) search
     * in optimistic search mode, regardless of the results returned by the first approximate search.
     * <p>
     * This flag should never be enabled in production; it is intended for testing and debugging only.
     */
    private static final boolean FORCE_REENTER_TESTING;

    static {
        FORCE_REENTER_TESTING = Boolean.parseBoolean(System.getProperty("mem_opt_srch.force_reenter", "false"));
    }

    private final KNNQuery knnQuery;
    private final QueryUtils queryUtils;
    private final boolean expandNestedDocs;

    @Override
    public Weight createWeight(IndexSearcher indexSearcher, ScoreMode scoreMode, float boost) throws IOException {
        final IndexReader reader = indexSearcher.getIndexReader();
        QueryProfiler profiler = KNNProfileUtil.getProfiler(indexSearcher);
        final KNNWeight knnWeight;
        if (profiler != null) {
            // add a new node to the profile tree
            profiler.getQueryBreakdown(knnQuery);
            knnWeight = (KNNWeight) knnQuery.createWeight(indexSearcher, scoreMode, 1);
            profiler.pollLastElement();
        } else {
            knnWeight = (KNNWeight) knnQuery.createWeight(indexSearcher, scoreMode, 1);
        }
        List<LeafReaderContext> leafReaderContexts = reader.leaves();
        List<PerLeafResult> perLeafResults;
        RescoreContext rescoreContext = knnQuery.getRescoreContext();
        final int finalK = knnQuery.getK();
        if (rescoreContext == null || !rescoreContext.isRescoreEnabled()) {
            perLeafResults = doSearch(indexSearcher, leafReaderContexts, knnWeight, finalK);
        } else {
            boolean isShardLevelRescoringDisabled = KNNSettings.isShardLevelRescoringDisabledForDiskBasedVector(knnQuery.getIndexName());
            int dimension = knnQuery.getQueryVector().length;
            int firstPassK = rescoreContext.getFirstPassK(finalK, isShardLevelRescoringDisabled, dimension);
            perLeafResults = doSearch(indexSearcher, leafReaderContexts, knnWeight, firstPassK);
            if (isShardLevelRescoringDisabled == false) {
                ResultUtil.reduceToTopK(perLeafResults, firstPassK);
            }

            StopWatch stopWatch = new StopWatch().start();
            perLeafResults = doRescore(indexSearcher, leafReaderContexts, knnWeight, perLeafResults, finalK);
            long rescoreTime = stopWatch.stop().totalTime().millis();
            log.debug("Rescoring results took {} ms. oversampled k:{}, segments:{}", rescoreTime, firstPassK, leafReaderContexts.size());
        }
        ResultUtil.reduceToTopK(perLeafResults, finalK);

        if (expandNestedDocs) {
            StopWatch stopWatch = new StopWatch().start();
            perLeafResults = retrieveAll(indexSearcher, leafReaderContexts, knnWeight, perLeafResults, rescoreContext == null);
            long time_in_millis = stopWatch.stop().totalTime().millis();
            if (log.isDebugEnabled()) {
                long totalNestedDocs = perLeafResults.stream().mapToLong(perLeafResult -> perLeafResult.getResult().scoreDocs.length).sum();
                log.debug("Expanding of nested docs took {} ms. totalNestedDocs:{} ", time_in_millis, totalNestedDocs);
            }
        }

        TopDocs[] topDocs = new TopDocs[perLeafResults.size()];
        for (int i = 0; i < perLeafResults.size(); i++) {
            TopDocs leafTopDocs = perLeafResults.get(i).getResult();
            for (ScoreDoc scoreDoc : leafTopDocs.scoreDocs) {
                scoreDoc.doc += leafReaderContexts.get(i).docBase;
            }
            topDocs[i] = leafTopDocs;
        }

        TopDocs topK = TopDocs.merge(getTotalTopDoc(topDocs), topDocs);

        if (topK.scoreDocs.length == 0) {
            return new MatchNoDocsQuery().createWeight(indexSearcher, scoreMode, boost);
        }
        return queryUtils.createDocAndScoreQuery(reader, topK, knnWeight).createWeight(indexSearcher, scoreMode, boost);
    }

    /**
     * When expandNestedDocs is set to true, additional nested documents are retrieved.
     * As a result, the total number of documents will exceed k.
     * Instead of relying on the k value, we must count the total number of documents
     * to accurately determine how many are in topDocs.
     * The theoretical maximum value this method could return is Integer.MAX_VALUE,
     * as a single shard cannot have more documents than Integer.MAX_VALUE.
     *
     * @param topDocs the top documents
     * @return the total number of documents in the topDocs
     */
    private int getTotalTopDoc(TopDocs[] topDocs) {
        if (expandNestedDocs == false) {
            return knnQuery.getK();
        }

        int sum = 0;
        for (TopDocs topDoc : topDocs) {
            sum += topDoc.scoreDocs.length;
        }
        return sum;
    }

    /**
     * Gets all leaves from nested fields when expandNestedDocs is true.
     * @param indexSearcher
     * @param leafReaderContexts
     * @param knnWeight
     * @param perLeafResults
     * @param useQuantizedVectors
     * @return List of PerLeafResult
     * @throws IOException
     */
    private List<PerLeafResult> retrieveAll(
        final IndexSearcher indexSearcher,
        List<LeafReaderContext> leafReaderContexts,
        KNNWeight knnWeight,
        List<PerLeafResult> perLeafResults,
        boolean useQuantizedVectors
    ) throws IOException {
        List<Callable<PerLeafResult>> nestedQueryTasks = new ArrayList<>(leafReaderContexts.size());
        for (int i = 0; i < perLeafResults.size(); i++) {
            LeafReaderContext leafReaderContext = leafReaderContexts.get(i);
            QueryProfiler profiler = KNNProfileUtil.getProfiler(indexSearcher);
            int finalI = i;
            nestedQueryTasks.add(() -> {
                PerLeafResult result = retrieveLeafResult(leafReaderContext, knnWeight, perLeafResults, useQuantizedVectors, finalI);
                if (profiler != null) {
                    AbstractProfileBreakdown profile = ((ContextualProfileBreakdown) profiler.getProfileBreakdown(this)).context(
                        leafReaderContext
                    );
                    LongMetric metric = (LongMetric) profile.getMetric(KNNMetrics.NUM_NESTED_DOCS);
                    metric.setValue((long) result.getResult().scoreDocs.length);
                }
                return result;
            });
        }
        return indexSearcher.getTaskExecutor().invokeAll(nestedQueryTasks);
    }

    /**
     * Gets a single leaf when expandNestedDocs is true.
     * @param leafReaderContext
     * @param knnWeight
     * @param perLeafResults
     * @param useQuantizedVectors
     * @param finalI
     * @return single PerLeafResult
     * @throws IOException
     */
    private PerLeafResult retrieveLeafResult(
        LeafReaderContext leafReaderContext,
        KNNWeight knnWeight,
        List<PerLeafResult> perLeafResults,
        boolean useQuantizedVectors,
        int finalI
    ) throws IOException {
        PerLeafResult perLeafResult = perLeafResults.get(finalI);
        if (perLeafResult.getResult().scoreDocs.length == 0) {
            return perLeafResult;
        }
        Set<Integer> docIds = Arrays.stream(perLeafResult.getResult().scoreDocs).map(scoreDoc -> scoreDoc.doc).collect(Collectors.toSet());
        DocIdSetIterator allSiblings = queryUtils.getAllSiblings(
            leafReaderContext,
            docIds,
            knnQuery.getParentsFilter(),
            perLeafResult.getFilterBits()
        );

        // Build exact search context
        final ExactSearcher.ExactSearcherContext exactSearcherContext = ExactSearcher.ExactSearcherContext.builder()
            .matchedDocsIterator(allSiblings)
            .numberOfMatchedDocs(allSiblings.cost())
            // setting to false because in re-scoring we want to do exact search on full precision vectors
            .useQuantizedVectorsForSearch(useQuantizedVectors)
            .k((int) allSiblings.cost())
            .field(knnQuery.getField())
            .radius(knnQuery.getRadius())
            .floatQueryVector(knnQuery.getQueryVector())
            .byteQueryVector(knnQuery.getByteQueryVector())
            .isMemoryOptimizedSearchEnabled(knnQuery.isMemoryOptimizedSearch())
            .build();

        // Run exact search
        TopDocs rescoreResult = knnWeight.exactSearch(leafReaderContext, exactSearcherContext);

        // Pack it as a result and return
        return new PerLeafResult(
            perLeafResult.getFilterBits(),
            perLeafResult.getFilterBitsCardinality(),
            rescoreResult,
            PerLeafResult.SearchMode.EXACT_SEARCH
        );
    }

    private List<PerLeafResult> doSearch(
        final IndexSearcher indexSearcher,
        List<LeafReaderContext> leafReaderContexts,
        KNNWeight knnWeight,
        int k
    ) throws IOException {
        // Collect search tasks
        List<Callable<PerLeafResult>> tasks = new ArrayList<>(leafReaderContexts.size());
        for (LeafReaderContext leafReaderContext : leafReaderContexts) {
            tasks.add(() -> searchLeaf(leafReaderContext, knnWeight, k));
        }

        // Execute search tasks
        final List<PerLeafResult> perLeafResults = indexSearcher.getTaskExecutor().invokeAll(tasks);

        // For memory optimized search, it should kick off 2nd search if optimistic
        if (knnQuery.isMemoryOptimizedSearch() && perLeafResults.size() > 1) {
            log.debug(
                "Running second deep dive search in optimistic while memory optimized search is enabled. perLeafResults.size()={}",
                perLeafResults.size()
            );
            final StopWatch stopWatch = startStopWatch(log);
            reentrantSearch(perLeafResults, knnWeight, leafReaderContexts, k, indexSearcher);
            stopStopWatchAndLog(log, stopWatch, "2ndOptimisticSearch", knnQuery.getShardId(), "All Shards", knnQuery.getField());
        }

        return perLeafResults;
    }

    private void reentrantSearch(
        final List<PerLeafResult> perLeafResults,
        final KNNWeight knnWeight,
        final List<LeafReaderContext> leafReaderContexts,
        final int k,
        final IndexSearcher indexSearcher
    ) throws IOException {
        if ((knnWeight instanceof MemoryOptimizedKNNWeight) == false) {
            log.error(
                "Memory optimized search was enabled, but got ["
                    + (knnWeight == null ? "null" : knnWeight.getClass().getSimpleName())
                    + "], expected="
                    + MemoryOptimizedKNNWeight.class.getSimpleName()
            );
            return;
        }

        assert (perLeafResults.size() == leafReaderContexts.size());

        // Get memory optimized knn weight first, it's safe get it, we checked it already.
        final MemoryOptimizedKNNWeight memoryOptKNNWeight = (MemoryOptimizedKNNWeight) knnWeight;

        // How many results have we collected?
        int totalResults = 0;
        for (PerLeafResult perLeafResult : perLeafResults) {
            totalResults += perLeafResult.getResult().scoreDocs.length;
        }

        // If we got empty results, then return immediately
        if (totalResults == 0) {
            return;
        }

        // Start 2nd deep dive, and get the minimum bar.
        final float minTopKScore = OptimisticSearchStrategyUtils.findKthLargestScore(perLeafResults, knnQuery.getK(), totalResults);

        // Select candidate segments for 2nd search. Pick whatever segment returned all vectors whose score values are greater than `kth`
        // value in the merged results.
        final List<Callable<TopDocs>> secondDeepDiveTasks = new ArrayList<>();
        final List<Integer> contextIndices = new ArrayList<>();
        final Map<Integer, TopDocs> segmentOrdToResults = new HashMap<>();

        for (int i = 0; i < leafReaderContexts.size(); ++i) {
            final LeafReaderContext leafReaderContext = leafReaderContexts.get(i);
            final PerLeafResult perLeafResult = perLeafResults.get(i);
            final TopDocs perLeaf = perLeafResults.get(i).getResult();
            if (perLeaf.scoreDocs.length > 0 && perLeafResult.getSearchMode() == PerLeafResult.SearchMode.APPROXIMATE_SEARCH) {
                if (FORCE_REENTER_TESTING || perLeaf.scoreDocs[perLeaf.scoreDocs.length - 1].score >= minTopKScore) {
                    log.debug("Entering the second deep dive approximate search while FORCE_REENTER_TESTING={}", FORCE_REENTER_TESTING);
                    // For the target segment, save top results. Which will be used as seeds.
                    segmentOrdToResults.put(leafReaderContext.ord, perLeaf);

                    // All this leaf's hits are at or above the global topK min score; explore it further
                    secondDeepDiveTasks.add(
                        () -> knnWeight.approximateSearch(
                            leafReaderContext,
                            perLeafResult.getFilterBits(),
                            perLeafResult.getFilterBitsCardinality(),
                            knnQuery.getK()
                        )
                    );
                    contextIndices.add(i);
                }
            }
        }

        // Kick off 2nd search tasks
        if (secondDeepDiveTasks.isEmpty() == false) {
            final ReentrantKnnCollectorManager reentrantCollectorManager = new ReentrantKnnCollectorManager(
                new TopKnnCollectorManager(k, indexSearcher),
                segmentOrdToResults,
                knnQuery.getQueryVector(),
                knnQuery.getField()
            );

            // Make weight use reentrant collector manager
            memoryOptKNNWeight.setReentrantKNNCollectorManager(reentrantCollectorManager);

            final List<TopDocs> deepDiveTopDocs = indexSearcher.getTaskExecutor().invokeAll(secondDeepDiveTasks);

            // Override results for target context
            for (int i = 0; i < deepDiveTopDocs.size(); ++i) {
                // Override with the new results
                final TopDocs resultsFromDeepDive = deepDiveTopDocs.get(i);
                final PerLeafResult perLeafResult = perLeafResults.get(contextIndices.get(i));
                perLeafResult.setResult(resultsFromDeepDive);
            }
        }
    }

    private List<PerLeafResult> doRescore(
        final IndexSearcher indexSearcher,
        List<LeafReaderContext> leafReaderContexts,
        KNNWeight knnWeight,
        List<PerLeafResult> perLeafResults,
        int k
    ) throws IOException {
        List<Callable<PerLeafResult>> rescoreTasks = new ArrayList<>(leafReaderContexts.size());
        for (int i = 0; i < perLeafResults.size(); i++) {
            LeafReaderContext leafReaderContext = leafReaderContexts.get(i);
            int finalI = i;
            rescoreTasks.add(() -> {
                PerLeafResult perLeafeResult = perLeafResults.get(finalI);
                if (perLeafeResult.getResult().scoreDocs.length == 0) {
                    return perLeafeResult;
                }
                final Set<Integer> docIds = Arrays.stream(perLeafeResult.getResult().scoreDocs)
                    .map(scoreDoc -> scoreDoc.doc)
                    .collect(Collectors.toSet());
                DocIdSetIterator matchedDocs;
                if (knnQuery.getParentsFilter() != null) {
                    matchedDocs = queryUtils.getAllSiblings(
                        leafReaderContext,
                        docIds,
                        knnQuery.getParentsFilter(),
                        perLeafeResult.getFilterBits()
                    );
                } else {
                    matchedDocs = new TopDocsDISI(perLeafeResult.getResult());
                }
                final ExactSearcher.ExactSearcherContext exactSearcherContext = ExactSearcher.ExactSearcherContext.builder()
                    .matchedDocsIterator(matchedDocs)
                    .numberOfMatchedDocs(matchedDocs.cost())
                    // setting to false because in re-scoring we want to do exact search on full precision vectors
                    .useQuantizedVectorsForSearch(false)
                    .k(k)
                    .radius(knnQuery.getRadius())
                    .field(knnQuery.getField())
                    .floatQueryVector(knnQuery.getQueryVector())
                    .byteQueryVector(knnQuery.getByteQueryVector())
                    .isMemoryOptimizedSearchEnabled(knnQuery.isMemoryOptimizedSearch())
                    .parentsFilter(knnQuery.getParentsFilter())
                    .build();
                TopDocs rescoreResult = knnWeight.exactSearch(leafReaderContext, exactSearcherContext);
                return new PerLeafResult(
                    perLeafeResult.getFilterBits(),
                    perLeafeResult.getFilterBitsCardinality(),
                    rescoreResult,
                    PerLeafResult.SearchMode.EXACT_SEARCH
                );
            });
        }
        return indexSearcher.getTaskExecutor().invokeAll(rescoreTasks);
    }

    private PerLeafResult searchLeaf(LeafReaderContext ctx, KNNWeight queryWeight, int k) throws IOException {
        final PerLeafResult perLeafResult = queryWeight.searchLeaf(ctx, k);
        final Bits liveDocs = ctx.reader().getLiveDocs();
        if (liveDocs != null) {
            List<ScoreDoc> list = new ArrayList<>();
            for (ScoreDoc scoreDoc : perLeafResult.getResult().scoreDocs) {
                if (liveDocs.get(scoreDoc.doc)) {
                    list.add(scoreDoc);
                }
            }
            ScoreDoc[] filteredScoreDoc = list.toArray(new ScoreDoc[0]);
            TotalHits totalHits = new TotalHits(filteredScoreDoc.length, TotalHits.Relation.EQUAL_TO);
            perLeafResult.setResult(new TopDocs(totalHits, filteredScoreDoc));
        }
        return perLeafResult;
    }

    @Override
    public String toString(String field) {
        return this.getClass().getSimpleName() + "[" + field + "]..." + KNNQuery.class.getSimpleName() + "[" + knnQuery.toString() + "]";
    }

    @Override
    public void visit(QueryVisitor visitor) {
        visitor.visitLeaf(this);
    }

    @Override
    public boolean equals(Object obj) {
        if (!sameClassAs(obj)) {
            return false;
        }
        return knnQuery == ((NativeEngineKnnVectorQuery) obj).knnQuery;
    }

    @Override
    public int hashCode() {
        return Objects.hash(classHash(), knnQuery.hashCode());
    }
}
