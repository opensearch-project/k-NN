/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.exactsearch;

import com.google.common.base.Predicates;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.NonNull;
import lombok.Value;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.HitQueue;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopDocsCollector;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.search.join.BitSetProducer;
import org.apache.lucene.util.BitSet;
import org.opensearch.common.Nullable;
import org.opensearch.common.lucene.Lucene;
import org.opensearch.knn.common.FieldInfoExtractor;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.query.SegmentLevelQuantizationInfo;
import org.opensearch.knn.index.query.SegmentLevelQuantizationUtil;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.query.scorers.VectorScorerMode;
import org.opensearch.knn.index.query.scorers.VectorScorers;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesIterator;
import org.opensearch.knn.indices.ModelDao;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.function.Predicate;

@Log4j2
@AllArgsConstructor
public class ExactSearcher {

    private final ModelDao modelDao;

    /**
     * Execute an exact search on a subset of documents of a leaf
     *
     * @param leafReaderContext {@link LeafReaderContext}
     * @param context {@link ExactSearcherContext}
     * @return TopDocs containing the results of the search
     * @throws IOException exception during execution of exact search
     */
    public TopDocs searchLeaf(final LeafReaderContext leafReaderContext, final ExactSearcherContext context) throws IOException {
        final VectorScorer vectorScorer = createVectorScorer(leafReaderContext, context);
        if (vectorScorer == null) {
            return TopDocsCollector.EMPTY_TOPDOCS;
        }
        if (context.getRadius() != null) {
            return doRadialSearch(leafReaderContext, context, vectorScorer);
        }
        if (context.getMatchedDocsIterator() != null && context.numberOfMatchedDocs <= context.getK()) {
            return scoreAllDocs(vectorScorer);
        }
        return searchTopCandidates(vectorScorer, context.getK(), Predicates.alwaysTrue());
    }

    /**
     * Perform radial search by comparing scores with min score. Currently, FAISS from native engine supports radial search.
     * Hence, we assume that Radius from knnQuery is always distance, and we convert it to score since we do exact search uses scores
     * to filter out the documents that does not have given min score.
     * @param leafReaderContext {@link LeafReaderContext}
     * @param context {@link ExactSearcherContext}
     * @param vectorScorer {@link VectorScorer}
     * @return TopDocs containing the results of the search
     * @throws IOException exception raised by scorer during traversal
     */
    private TopDocs doRadialSearch(LeafReaderContext leafReaderContext, ExactSearcherContext context, VectorScorer vectorScorer)
        throws IOException {
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
        // For FAISS exact search, radius is in FAISS distance space (e.g., inner product for cosine).
        // We need to convert it to OpenSearch score space using the reverse translation.
        final float minScore = context.isMemoryOptimizedSearchEnabled ? context.getRadius() : engine.score(context.getRadius(), spaceType);

        return filterDocsByMinScore(context, vectorScorer, minScore);
    }

    private TopDocs scoreAllDocs(VectorScorer vectorScorer) throws IOException {
        final DocIdSetIterator iterator = vectorScorer.iterator();
        final List<ScoreDoc> scoreDocList = new ArrayList<>();
        int docId;
        while ((docId = iterator.nextDoc()) != DocIdSetIterator.NO_MORE_DOCS) {
            scoreDocList.add(new ScoreDoc(docId, vectorScorer.score()));
        }
        scoreDocList.sort(Comparator.comparing(scoreDoc -> scoreDoc.score, Comparator.reverseOrder()));
        return new TopDocs(new TotalHits(scoreDocList.size(), TotalHits.Relation.EQUAL_TO), scoreDocList.toArray(ScoreDoc[]::new));
    }

    private TopDocs searchTopCandidates(VectorScorer vectorScorer, int limit, @NonNull Predicate<Float> filterScore) throws IOException {
        final DocIdSetIterator iterator = vectorScorer.iterator();
        // Creating min heap and init with MAX DocID and Score as -INF.
        final HitQueue queue = new HitQueue(limit, true);
        ScoreDoc topDoc = queue.top();
        int docId;
        while ((docId = iterator.nextDoc()) != DocIdSetIterator.NO_MORE_DOCS) {
            final float currentScore = vectorScorer.score();
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

    private TopDocs filterDocsByMinScore(ExactSearcherContext context, VectorScorer vectorScorer, float minScore) throws IOException {
        int maxResultWindow = context.getMaxResultWindow();
        Predicate<Float> scoreGreaterThanOrEqualToMinScore = score -> score >= minScore;
        return searchTopCandidates(vectorScorer, maxResultWindow, scoreGreaterThanOrEqualToMinScore);
    }

    private VectorScorer createVectorScorer(final LeafReaderContext leafReaderContext, final ExactSearcherContext context)
        throws IOException {
        final SegmentReader reader = Lucene.segmentReader(leafReaderContext.reader());
        final FieldInfo fieldInfo = FieldInfoExtractor.getFieldInfo(reader, context.getField());
        if (fieldInfo == null) {
            log.debug("[KNN] Cannot create VectorScorer as FieldInfo not found for {}:{}", context.getField(), reader.getSegmentName());
            return null;
        }

        final VectorDataType vectorDataType = FieldInfoExtractor.extractVectorDataType(fieldInfo);
        final SpaceType spaceType = FieldInfoExtractor.getSpaceType(modelDao, fieldInfo);
        final VectorScorerMode scorerMode = context.isUseQuantizedVectorsForSearch() ? VectorScorerMode.SCORE : VectorScorerMode.RESCORE;
        final boolean isNestedRequired = context.getParentsFilter() != null;
        final DocIdSetIterator acceptedChildrenIterator = isNestedRequired ? context.getMatchedDocsIterator() : null;
        final BitSet parentBitSet = isNestedRequired ? context.getParentsFilter().getBitSet(leafReaderContext) : null;

        final KNNVectorValues<?> vectorValues = KNNVectorValuesFactory.getVectorValues(fieldInfo, reader);
        final KNNVectorValuesIterator.DocIdsIteratorValues iteratorValues = (KNNVectorValuesIterator.DocIdsIteratorValues) vectorValues
            .getVectorValuesIterator();

        if (VectorDataType.BINARY == vectorDataType) {
            return VectorScorers.createScorer(
                iteratorValues,
                context.getByteQueryVector(),
                scorerMode,
                spaceType,
                acceptedChildrenIterator,
                parentBitSet
            );
        }

        if (VectorDataType.BYTE == vectorDataType) {
            final float[] floatQueryVector = context.getFloatQueryVector();
            final byte[] byteQueryVector = new byte[floatQueryVector.length];
            for (int i = 0; i < byteQueryVector.length; i++) {
                byteQueryVector[i] = (byte) floatQueryVector[i];
            }
            return VectorScorers.createScorer(
                iteratorValues,
                byteQueryVector,
                scorerMode,
                spaceType,
                acceptedChildrenIterator,
                parentBitSet
            );
        }

        // Float vector path
        final SegmentLevelQuantizationInfo quantizationInfo = SegmentLevelQuantizationInfo.build(reader, fieldInfo, context.getField());

        if (quantizationInfo == null || scorerMode == VectorScorerMode.RESCORE) {
            return VectorScorers.createScorer(
                iteratorValues,
                context.getFloatQueryVector(),
                scorerMode,
                spaceType,
                fieldInfo,
                acceptedChildrenIterator,
                parentBitSet
            );
        }

        // Quantized path — need byte vector values
        final KNNVectorValues<?> quantizedValues = KNNVectorValuesFactory.getVectorValues(fieldInfo, reader, true);
        final KNNVectorValuesIterator.DocIdsIteratorValues quantizedIteratorValues =
            (KNNVectorValuesIterator.DocIdsIteratorValues) quantizedValues.getVectorValuesIterator();

        if (SegmentLevelQuantizationUtil.isAdcEnabled(quantizationInfo)) {
            SegmentLevelQuantizationUtil.transformVectorWithADC(context.getFloatQueryVector(), quantizationInfo, spaceType);
            return VectorScorers.createScorer(
                quantizedIteratorValues,
                context.getFloatQueryVector(),
                scorerMode,
                spaceType,
                fieldInfo,
                acceptedChildrenIterator,
                parentBitSet
            );
        }

        final byte[] quantizedQueryVector = SegmentLevelQuantizationUtil.quantizeVector(context.getFloatQueryVector(), quantizationInfo);
        return VectorScorers.createScorer(
            quantizedIteratorValues,
            quantizedQueryVector,
            scorerMode,
            SpaceType.HAMMING,
            acceptedChildrenIterator,
            parentBitSet
        );
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
        @Nullable
        DocIdSetIterator matchedDocsIterator;
        long numberOfMatchedDocs;
        /**
         * whether the matchedDocs contains parent ids or child ids. This is relevant in the case of
         * filtered nested search where the matchedDocs contain the parent ids and {@link NestedVectorIdsExactKNNIterator}
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
