/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import lombok.Builder;
import lombok.NonNull;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchNoDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopDocsCollector;
import org.apache.lucene.search.Weight;
import org.opensearch.common.Nullable;
import org.opensearch.common.StopWatch;
import org.opensearch.knn.index.query.common.QueryUtils;
import org.opensearch.knn.indices.ModelDao;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.Callable;

@Builder
@Log4j2
public class RescoreKNNVectorQuery extends Query {

    @NonNull
    private Query innerQuery;
    @NonNull
    private String field;
    private int k;
    @NonNull
    private QueryVector queryVector;
    @NonNull
    private QueryUtils queryUtils;

    // Note: ideally query should not have to deal with shard level information. Adding it for logging purposes only
    // TODO: ThreadContext does not work with logger, remove this from here once its figured out
    private int shardId;

    @Override
    public Weight createWeight(IndexSearcher searcher, ScoreMode scoreMode, float boost) throws IOException {
        Weight weight = searcher.createWeight(innerQuery, scoreMode, boost);
        StopWatch stopWatch = startStopWatch();
        TopDocs[] perLeafResults = doRescore(searcher, weight, k);
        stopStopWatchAndLog(stopWatch);
        TopDocs topK = TopDocs.merge(k, perLeafResults);
        if (topK.scoreDocs.length == 0) {
            return new MatchNoDocsQuery().createWeight(searcher, scoreMode, boost);
        }
        return queryUtils.createDocAndScoreQuery(searcher.getIndexReader(), topK).createWeight(searcher, scoreMode, boost);
    }

    private TopDocs[] doRescore(final IndexSearcher indexSearcher, Weight weight, int k) throws IOException {
        List<LeafReaderContext> leafReaderContexts = indexSearcher.getIndexReader().leaves();
        List<Callable<TopDocs>> rescoreTasks = new ArrayList<>(leafReaderContexts.size());
        ExactSearcher searcher = new ExactSearcher(ModelDao.OpenSearchKNNModelDao.getInstance());
        for (LeafReaderContext leafReaderContext : leafReaderContexts) {
            rescoreTasks.add(() -> searchLeaf(searcher, weight, k, leafReaderContext));
        }
        return indexSearcher.getTaskExecutor().invokeAll(rescoreTasks).toArray(TopDocs[]::new);
    }

    private TopDocs searchLeaf(ExactSearcher searcher, Weight weight, int k, LeafReaderContext leafReaderContext) throws IOException {
        Scorer scorer = weight.scorer(leafReaderContext);
        if (scorer == null) {
            return TopDocsCollector.EMPTY_TOPDOCS;
        }
        DocIdSetIterator iterator = scorer.iterator();
        final ExactSearcher.ExactSearcherContext exactSearcherContext = ExactSearcher.ExactSearcherContext.builder()
            .matchedDocsIterator(iterator)
            .numberOfMatchedDocs(iterator.cost())
            // setting to false because in re-scoring we want to do exact search on full precision vectors
            .useQuantizedVectorsForSearch(false)
            .k(k)
            .field(field)
            .queryVector(queryVector)
            .build();
        Map<Integer, Float> integerFloatMap = searcher.searchLeaf(leafReaderContext, exactSearcherContext);
        return ResultUtil.resultMapToTopDocs(integerFloatMap, leafReaderContext.docBase);
    }

    private StopWatch startStopWatch() {
        if (log.isDebugEnabled()) {
            return new StopWatch().start();
        }
        return null;
    }

    private void stopStopWatchAndLog(@Nullable final StopWatch stopWatch) {
        if (log.isDebugEnabled() && stopWatch != null) {
            stopWatch.stop();
            log.debug(
                "[{}] shard: [{}], field: [{}], time in nanos:[{}] ",
                this.getClass().getSimpleName(),
                shardId,
                field,
                stopWatch.totalTime().nanos()
            );
        }
    }

    @Override
    public String toString(String field) {
        return this.getClass().getSimpleName()
            + "innerQuery="
            + innerQuery
            + "field="
            + field
            + ", vector="
            + queryVector
            + ", k="
            + k
            + ", shardId="
            + shardId
            + "]";
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
        RescoreKNNVectorQuery other = (RescoreKNNVectorQuery) obj;
        return Objects.equals(innerQuery, other.innerQuery)
            && Objects.equals(queryVector, other.queryVector)
            && Objects.equals(field, other.field)
            && k == other.k
            && shardId == other.shardId;
    }

    @Override
    public int hashCode() {
        return Objects.hash(innerQuery, queryVector, field, k, shardId);
    }
}
