/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import com.google.common.annotations.VisibleForTesting;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchNoDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopDocsCollector;
import org.apache.lucene.search.Weight;
import org.opensearch.common.Nullable;
import org.opensearch.common.StopWatch;
import org.opensearch.knn.index.query.common.QueryUtils;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.knn.profile.KNNProfileUtil;
import org.opensearch.knn.profile.query.KNNQueryTimingType;
import org.opensearch.search.profile.ContextualProfileBreakdown;
import org.opensearch.search.profile.query.QueryProfiler;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.Callable;

@Log4j2
public class RescoreKNNVectorQuery extends Query {

    private final Query innerQuery;
    private final String field;
    private final int k;
    private final float[] queryVector;
    // Note: ideally query should not have to deal with shard level information. Adding it for logging purposes only
    // TODO: ThreadContext does not work with logger, remove this from here once its figured out
    private final int shardId;
    private final ExactSearcher exactSearcher;

    /**
     * Constructs a new RescoreKNNVectorQuery.
     *
     * @param innerQuery  The initial query to execute before rescoring
     * @param field      The field name containing the vector data
     * @param k          The number of nearest neighbors to return
     * @param queryVector The vector to compare against document vectors
     */
    public RescoreKNNVectorQuery(Query innerQuery, String field, int k, float[] queryVector, int shardId) {
        this.innerQuery = innerQuery;
        this.field = field;
        this.k = k;
        this.queryVector = queryVector;
        this.shardId = shardId;
        this.exactSearcher = new ExactSearcher(ModelDao.OpenSearchKNNModelDao.getInstance());
    }

    @VisibleForTesting
    public RescoreKNNVectorQuery(Query innerQuery, String field, int k, float[] queryVector, int shardId, ExactSearcher searcher) {
        this.innerQuery = innerQuery;
        this.field = field;
        this.k = k;
        this.queryVector = queryVector;
        this.shardId = shardId;
        this.exactSearcher = searcher;
    }

    @Override
    public Weight createWeight(IndexSearcher searcher, ScoreMode scoreMode, float boost) throws IOException {
        final Query rewrittenInnerQuery = searcher.rewrite(innerQuery);
        final Weight weight = searcher.createWeight(rewrittenInnerQuery, scoreMode, boost);
        final StopWatch stopWatch = startStopWatch();
        final TopDocs[] perLeafResults = doRescore(searcher, weight);
        stopStopWatchAndLog(stopWatch);
        final TopDocs topK = TopDocs.merge(k, perLeafResults);
        if (topK.scoreDocs.length == 0) {
            return new MatchNoDocsQuery().createWeight(searcher, scoreMode, boost);
        }
        return QueryUtils.getInstance().createDocAndScoreQuery(searcher.getIndexReader(), topK).createWeight(searcher, scoreMode, boost);
    }

    private TopDocs[] doRescore(final IndexSearcher indexSearcher, Weight weight) throws IOException {
        List<LeafReaderContext> leafReaderContexts = indexSearcher.getIndexReader().leaves();
        List<Callable<TopDocs>> rescoreTasks = new ArrayList<>(leafReaderContexts.size());
        QueryProfiler profiler = KNNProfileUtil.getProfiler(indexSearcher);
        ContextualProfileBreakdown profile;
        if (profiler != null) {
            profile = (ContextualProfileBreakdown) profiler.getProfileBreakdown(this);
        } else {
            profile = null;
        }
        for (LeafReaderContext leafReaderContext : leafReaderContexts) {
            rescoreTasks.add(() -> searchLeaf(exactSearcher, weight, k, leafReaderContext, profile));
        }
        return indexSearcher.getTaskExecutor().invokeAll(rescoreTasks).toArray(TopDocs[]::new);
    }

    private TopDocs searchLeaf(
        ExactSearcher searcher,
        Weight weight,
        int k,
        LeafReaderContext leafReaderContext,
        ContextualProfileBreakdown profile
    ) throws IOException {
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
            .floatQueryVector(queryVector)
            .build();
        TopDocs results = (TopDocs) KNNProfileUtil.profile(profile, leafReaderContext, KNNQueryTimingType.EXACT_SEARCH, () -> {
            try {
                return searcher.searchLeaf(leafReaderContext, exactSearcherContext);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });
        if (leafReaderContext.docBase > 0) {
            for (ScoreDoc scoreDoc : results.scoreDocs) {
                scoreDoc.doc += leafReaderContext.docBase;
            }
        }
        return results;
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
