/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import com.google.common.base.Predicates;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.NonNull;
import lombok.Setter;
import lombok.Value;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.HitQueue;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TaskExecutor;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopDocsCollector;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.search.join.BitSetProducer;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.BitSetIterator;
import org.opensearch.common.lucene.Lucene;
import org.opensearch.knn.common.FieldInfoExtractor;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.query.iterators.BinaryVectorIdsKNNIterator;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.query.iterators.ByteVectorIdsKNNIterator;
import org.opensearch.knn.index.query.iterators.NestedBinaryVectorIdsKNNIterator;
import org.opensearch.knn.index.query.iterators.RangeDocIdSetIterator;
import org.opensearch.knn.index.query.iterators.VectorIdsKNNIterator;
import org.opensearch.knn.index.query.iterators.KNNIterator;
import org.opensearch.knn.index.query.iterators.NestedByteVectorIdsKNNIterator;
import org.opensearch.knn.index.query.iterators.NestedVectorIdsKNNIterator;
import org.opensearch.knn.index.vectorvalues.KNNBinaryVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNByteVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNFloatVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.threadpool.ThreadPool;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.Callable;
import java.util.function.Predicate;

import static org.apache.lucene.index.IndexWriter.MAX_DOCS;
import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;
import static org.opensearch.knn.common.KNNConstants.EXACT_SEARCH_THREAD_POOL;

@Log4j2
@AllArgsConstructor
public class ExactSearcher {

    private final ModelDao modelDao;
    private static TaskExecutor taskExecutor;

    @Setter
    private static boolean concurrentExactSearchEnabled = KNNSettings.KNN_DEFAULT_CONCURRENT_EXACT_SEARCH_ENABLED;
    @Setter
    private static int concurrentExactSearchMaxPartitionCount = KNNSettings.KNN_DEFAULT_CONCURRENT_EXACT_SEARCH_MAX_PARTITION_COUNT;
    @Setter
    private static int concurrentExactSearchMinDocumentCount = KNNSettings.KNN_DEFAULT_CONCURRENT_EXACT_SEARCH_MIN_DOCUMENT_COUNT;

    public static void initialize(ThreadPool threadPool) {
        ExactSearcher.taskExecutor = new TaskExecutor(threadPool.executor(EXACT_SEARCH_THREAD_POOL));
    }

    /**
     * Execute an exact search on a leaf
     *
     * @param leafReaderContext {@link LeafReaderContext}
     * @param context {@link ExactSearcherContext}
     * @return TopDocs containing the results of the search
     * @throws IOException exception during execution of exact search
     */
    public TopDocs searchLeaf(final LeafReaderContext leafReaderContext, final ExactSearcherContext context) throws IOException {
        if (context.getRadius() != null) {
            return doRadialSearch(leafReaderContext, context);
        }
        if (context.getMatchedDocs() != null && context.numberOfMatchedDocs <= context.getK()) {
            return scoreAllDocs(leafReaderContext, context);
        }
        return partitionLeaf(leafReaderContext, context, context.getK(), Predicates.alwaysTrue());
    }

    /**
     * Perform radial search by comparing scores with min score. Currently, FAISS from native engine supports radial search.
     * Hence, we assume that Radius from knnQuery is always distance, and we convert it to score since we do exact search uses scores
     * to filter out the documents that does not have given min score.
     * @param leafReaderContext {@link LeafReaderContext}
     * @param context {@link ExactSearcherContext}
     * @return TopDocs containing the results of the search
     * @throws IOException exception raised by iterator during traversal
     */
    private TopDocs doRadialSearch(LeafReaderContext leafReaderContext, ExactSearcherContext context) throws IOException {
        // Ensure `isMemoryOptimizedSearchEnabled` is set. This is necessary to determine whether distance to score conversion is required.
        assert (context.isMemoryOptimizedSearchEnabled != null);

        final SegmentReader reader = Lucene.segmentReader(leafReaderContext.reader());
        final FieldInfo fieldInfo = FieldInfoExtractor.getFieldInfo(reader, context.getField());
        if (fieldInfo == null) {
            return TopDocsCollector.EMPTY_TOPDOCS;
        }
        final KNNEngine engine = FieldInfoExtractor.extractKNNEngine(fieldInfo);
        if (KNNEngine.FAISS != engine) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "Engine [%s] does not support radial search", engine));
        }
        final SpaceType spaceType = FieldInfoExtractor.getSpaceType(modelDao, fieldInfo);
        // We need to use the given radius when memory optimized search is enabled. Since it relies on Lucene's scoring framework, the given
        // max distance is already converted min score then saved in `radius`. Thus, we don't need a score translation which does not make
        // sense as it is treating min score as a max distance otherwise.
        final float minScore = context.isMemoryOptimizedSearchEnabled
            ? context.getRadius()
            : spaceType.scoreTranslation(context.getRadius());

        return filterDocsByMinScore(leafReaderContext, context, minScore);
    }

    private TopDocs filterDocsByMinScore(LeafReaderContext leafReaderContext, ExactSearcherContext context, float minScore)
        throws IOException {
        int maxResultWindow = context.getMaxResultWindow();
        Predicate<Float> scoreGreaterThanOrEqualToMinScore = score -> score >= minScore;
        return partitionLeaf(leafReaderContext, context, maxResultWindow, scoreGreaterThanOrEqualToMinScore);
    }

    private TopDocs partitionLeaf(
        LeafReaderContext leafReaderContext,
        ExactSearcherContext context,
        int limit,
        @NonNull Predicate<Float> filterScore
    ) throws IOException {
        int maxDoc = leafReaderContext.reader().maxDoc();
        BitSetProducer parentsFilter = context.getParentsFilter();
        BitSet parentBitSet = parentsFilter != null ? parentsFilter.getBitSet(leafReaderContext) : null;
        int partitions = computeNumPartitions(maxDoc);
        if (partitions == 1) {
            return searchTopCandidates(leafReaderContext, context, limit, filterScore, 0, maxDoc);
        }
        List<Callable<TopDocs>> tasks = new ArrayList<>(partitions);
        int baseSize = maxDoc / partitions, remainder = maxDoc % partitions;
        int offset = 0;
        for (int i = 0; i < partitions; i++) {
            int size = baseSize + (i < remainder ? 1 : 0);
            int minDocId = offset, maxDocId = computePartitionEnd(parentBitSet, offset + size, maxDoc);
            if (minDocId >= maxDocId) break;
            tasks.add(() -> searchTopCandidates(leafReaderContext, context, limit, filterScore, minDocId, maxDocId));
            offset = maxDocId;
        }
        List<TopDocs> results = ExactSearcher.taskExecutor.invokeAll(tasks);
        return TopDocs.merge(limit, results.toArray(new TopDocs[tasks.size()]));
    }

    private int computeNumPartitions(int maxDoc) {
        if (!concurrentExactSearchEnabled) {
            return 1;
        }
        int partitions = Math.max(maxDoc / concurrentExactSearchMinDocumentCount, 1);
        return (concurrentExactSearchMaxPartitionCount == 0) ? partitions : Math.min(partitions, concurrentExactSearchMaxPartitionCount);
    }

    private int computePartitionEnd(BitSet parentBitSet, int tentativeEnd, int maxDoc) {
        if (parentBitSet == null) {
            return tentativeEnd;
        }
        if (tentativeEnd >= maxDoc) {
            return maxDoc;
        }
        int nextParent = parentBitSet.nextSetBit(tentativeEnd);
        return (nextParent == NO_MORE_DOCS) ? maxDoc : nextParent + 1;
    }

    private TopDocs searchTopCandidates(LeafReaderContext leafReaderContext, ExactSearcherContext context, int limit, @NonNull Predicate<Float> filterScore, int minDocId, int maxDocId) throws IOException {
        DocIdSetIterator matchedDocsIterator = context.getMatchedDocs() != null
                ? new RangeDocIdSetIterator(new BitSetIterator(context.getMatchedDocs(), maxDocId - minDocId), minDocId, maxDocId)
                : DocIdSetIterator.range(minDocId, maxDocId);
        KNNIterator iterator = getKNNIterator(leafReaderContext, context, matchedDocsIterator);
        if (iterator == null) {
            return TopDocsCollector.EMPTY_TOPDOCS;
        }
        // Creating min heap and init with MAX DocID and Score as -INF.
        final HitQueue queue = new HitQueue(limit, true);
        ScoreDoc topDoc = queue.top();
        int docId;
        while ((docId = iterator.nextDoc()) != NO_MORE_DOCS) {
            final float currentScore = iterator.score();
            if (filterScore.test(currentScore) && currentScore > topDoc.score) {
                topDoc.score = currentScore;
                topDoc.doc = docId;
                // As the HitQueue is min heap, updating top will bring the doc with -INF score or worst score we
                // have seen till now on top.
                topDoc = queue.updateTop();
            }
        }

        // If scores are negative we will remove them.
        // This is done, because there can be negative values in the Heap as we init the heap with Score as -INF.
        // If filterIds < k, some values in heap can have a negative score.
        while (queue.size() > 0 && queue.top().score < 0) {
            queue.pop();
        }

        ScoreDoc[] topScoreDocs = new ScoreDoc[queue.size()];
        for (int i = topScoreDocs.length - 1; i >= 0; i--) {
            topScoreDocs[i] = queue.pop();
        }

        TotalHits totalHits = new TotalHits(topScoreDocs.length, TotalHits.Relation.EQUAL_TO);
        return new TopDocs(totalHits, topScoreDocs);
    }

    private KNNIterator getKNNIterator(
        LeafReaderContext leafReaderContext,
        ExactSearcherContext exactSearcherContext,
        DocIdSetIterator matchedDocsIterator
    ) throws IOException {
        final SegmentReader reader = Lucene.segmentReader(leafReaderContext.reader());
        final FieldInfo fieldInfo = FieldInfoExtractor.getFieldInfo(reader, exactSearcherContext.getField());
        if (fieldInfo == null) {
            log.debug(
                "[KNN] Cannot get KNNIterator as Field info not found for {}:{}",
                exactSearcherContext.getField(),
                reader.getSegmentName()
            );
            return null;
        }
        final VectorDataType vectorDataType = FieldInfoExtractor.extractVectorDataType(fieldInfo);
        final SpaceType spaceType = FieldInfoExtractor.getSpaceType(modelDao, fieldInfo);
        boolean isNestedRequired = exactSearcherContext.getParentsFilter() != null;

        if (VectorDataType.BINARY == vectorDataType) {
            final KNNVectorValues<byte[]> vectorValues = KNNVectorValuesFactory.getVectorValues(fieldInfo, reader);
            if (isNestedRequired) {
                return new NestedBinaryVectorIdsKNNIterator(
                    matchedDocsIterator,
                    exactSearcherContext.getByteQueryVector(),
                    (KNNBinaryVectorValues) vectorValues,
                    spaceType,
                    exactSearcherContext.getParentsFilter().getBitSet(leafReaderContext)
                );
            }
            return new BinaryVectorIdsKNNIterator(
                matchedDocsIterator,
                exactSearcherContext.getByteQueryVector(),
                (KNNBinaryVectorValues) vectorValues,
                spaceType
            );
        }

        if (VectorDataType.BYTE == vectorDataType) {
            final KNNVectorValues<byte[]> vectorValues = KNNVectorValuesFactory.getVectorValues(fieldInfo, reader);
            if (isNestedRequired) {
                return new NestedByteVectorIdsKNNIterator(
                    matchedDocsIterator,
                    exactSearcherContext.getFloatQueryVector(),
                    (KNNByteVectorValues) vectorValues,
                    spaceType,
                    exactSearcherContext.getParentsFilter().getBitSet(leafReaderContext)
                );
            }
            return new ByteVectorIdsKNNIterator(
                matchedDocsIterator,
                exactSearcherContext.getFloatQueryVector(),
                (KNNByteVectorValues) vectorValues,
                spaceType
            );
        }
        byte[] quantizedQueryVector = null;
        SegmentLevelQuantizationInfo segmentLevelQuantizationInfo = null;
        if (exactSearcherContext.isUseQuantizedVectorsForSearch()) {
            // Build Segment Level Quantization info.
            segmentLevelQuantizationInfo = SegmentLevelQuantizationInfo.build(
                reader,
                fieldInfo,
                exactSearcherContext.getField(),
                reader.getSegmentInfo().info.getVersion()
            );
            // Quantize the Query Vector Once. Or transform it in the case of ADC.
            if (SegmentLevelQuantizationUtil.isAdcEnabled(segmentLevelQuantizationInfo)) {
                SegmentLevelQuantizationUtil.transformVectorWithADC(
                    exactSearcherContext.getFloatQueryVector(),
                    segmentLevelQuantizationInfo,
                    spaceType
                );
            } else {
                quantizedQueryVector = SegmentLevelQuantizationUtil.quantizeVector(
                    exactSearcherContext.getFloatQueryVector(),
                    segmentLevelQuantizationInfo
                );
            }
        }

        final KNNVectorValues<float[]> vectorValues = KNNVectorValuesFactory.getVectorValues(fieldInfo, reader);
        if (isNestedRequired) {
            return new NestedVectorIdsKNNIterator(
                matchedDocsIterator,
                exactSearcherContext.getFloatQueryVector(),
                (KNNFloatVectorValues) vectorValues,
                spaceType,
                exactSearcherContext.getParentsFilter().getBitSet(leafReaderContext),
                quantizedQueryVector,
                segmentLevelQuantizationInfo
            );
        }
        return new VectorIdsKNNIterator(
            matchedDocsIterator,
            exactSearcherContext.getFloatQueryVector(),
            (KNNFloatVectorValues) vectorValues,
            spaceType,
            quantizedQueryVector,
            segmentLevelQuantizationInfo
        );
    }

    private TopDocs scoreAllDocs(LeafReaderContext leafReaderContext, ExactSearcherContext exactSearcherContext) throws IOException {
        BitSet matchedDocs = exactSearcherContext.getMatchedDocs();
        DocIdSetIterator matchedDocsIterator = matchedDocs != null
            ? new BitSetIterator(matchedDocs, exactSearcherContext.getNumberOfMatchedDocs())
            : DocIdSetIterator.range(0, MAX_DOCS);
        KNNIterator iterator = getKNNIterator(leafReaderContext, exactSearcherContext, matchedDocsIterator);
        final List<ScoreDoc> scoreDocList = new ArrayList<>();
        int docId;
        while ((docId = iterator.nextDoc()) != NO_MORE_DOCS) {
            scoreDocList.add(new ScoreDoc(docId, iterator.score()));
        }
        scoreDocList.sort(Comparator.comparing(scoreDoc -> scoreDoc.score, Comparator.reverseOrder()));
        return new TopDocs(new TotalHits(scoreDocList.size(), TotalHits.Relation.EQUAL_TO), scoreDocList.toArray(ScoreDoc[]::new));
    }

    /**
     * Stores the context that is used to do the exact search. This class will help in reducing the explosion of attributes
     * for doing exact search.
     */
    @Value
    @Builder
    public static class ExactSearcherContext {
        /**
         * controls whether we should use Quantized vectors during exact search or not. This is useful because when we do
         * re-scoring we need to re-score using full precision vectors and not quantized vectors.
         */
        boolean useQuantizedVectorsForSearch;
        int k;
        Float radius;
        BitSet matchedDocs;
        long numberOfMatchedDocs;
        /**
         * whether the matchedDocs contains parent ids or child ids. This is relevant in the case of
         * filtered nested search where the matchedDocs contain the parent ids and {@link NestedVectorIdsKNNIterator}
         * needs to be used.
         */
        BitSetProducer parentsFilter;
        float[] floatQueryVector;
        byte[] byteQueryVector;
        String field;
        Integer maxResultWindow;
        VectorSimilarityFunction similarityFunction;
        Boolean isMemoryOptimizedSearchEnabled;
    }
}
