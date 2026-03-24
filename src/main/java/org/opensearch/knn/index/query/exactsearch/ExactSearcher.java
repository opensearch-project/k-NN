/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.exactsearch;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Value;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocAndFloatFeatureBuffer;
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
        // When nested, matchedDocsIterator is already consumed inside NestedBestChildVectorScorer,
        // so pass null to avoid double consumption of the same iterator.
        final boolean isNested = context.getParentsFilter() != null;
        final DocIdSetIterator matchedDocs = isNested ? null : context.getMatchedDocsIterator();
        if (context.getRadius() != null) {
            return doRadialSearch(leafReaderContext, context, vectorScorer, matchedDocs);
        }
        if (matchedDocs != null && context.numberOfMatchedDocs <= context.getK()) {
            return scoreAllDocs(vectorScorer, matchedDocs);
        }
        return searchTopK(vectorScorer, matchedDocs, context.getK());
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
    private TopDocs doRadialSearch(
        final LeafReaderContext leafReaderContext,
        final ExactSearcherContext context,
        final VectorScorer vectorScorer,
        final DocIdSetIterator matchedDocs
    ) throws IOException {
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
        final float minScore = context.isMemoryOptimizedSearchEnabled ? context.getRadius() : engine.score(context.getRadius(), spaceType);

        return searchWithMinScore(vectorScorer, matchedDocs, context.getMaxResultWindow(), minScore);
    }

    private TopDocs scoreAllDocs(final VectorScorer vectorScorer, final DocIdSetIterator matchedDocs) throws IOException {
        final VectorScorer.Bulk bulkScorer = vectorScorer.bulk(matchedDocs);
        final DocAndFloatFeatureBuffer buffer = new DocAndFloatFeatureBuffer();
        final List<ScoreDoc> scoreDocList = new ArrayList<>();

        while (true) {
            bulkScorer.nextDocsAndScores(DocIdSetIterator.NO_MORE_DOCS, null, buffer);
            if (buffer.size == 0) {
                break;
            }
            for (int i = 0; i < buffer.size; i++) {
                scoreDocList.add(new ScoreDoc(buffer.docs[i], buffer.features[i]));
            }
        }

        scoreDocList.sort(Comparator.comparing(scoreDoc -> scoreDoc.score, Comparator.reverseOrder()));
        return new TopDocs(new TotalHits(scoreDocList.size(), TotalHits.Relation.EQUAL_TO), scoreDocList.toArray(ScoreDoc[]::new));
    }

    private TopDocs searchTopK(final VectorScorer vectorScorer, final DocIdSetIterator matchedDocs, final int k) throws IOException {
        final VectorScorer.Bulk bulkScorer = vectorScorer.bulk(matchedDocs);
        final DocAndFloatFeatureBuffer buffer = new DocAndFloatFeatureBuffer();
        final HitQueue queue = new HitQueue(k, true);
        ScoreDoc topDoc = queue.top();

        for (float maxScore = bulkScorer.nextDocsAndScores(DocIdSetIterator.NO_MORE_DOCS, null, buffer); buffer.size > 0; maxScore =
            bulkScorer.nextDocsAndScores(DocIdSetIterator.NO_MORE_DOCS, null, buffer)) {
            if (maxScore < topDoc.score) {
                continue;
            }
            for (int i = 0; i < buffer.size; i++) {
                if (buffer.features[i] > topDoc.score) {
                    topDoc.score = buffer.features[i];
                    topDoc.doc = buffer.docs[i];
                    topDoc = queue.updateTop();
                }
            }
        }

        while (queue.size() > 0 && queue.top().score < 0) {
            queue.pop();
        }

        final ScoreDoc[] topScoreDocs = new ScoreDoc[queue.size()];
        for (int i = topScoreDocs.length - 1; i >= 0; i--) {
            topScoreDocs[i] = queue.pop();
        }
        return new TopDocs(new TotalHits(topScoreDocs.length, TotalHits.Relation.EQUAL_TO), topScoreDocs);
    }

    private TopDocs searchWithMinScore(
        final VectorScorer vectorScorer,
        final DocIdSetIterator matchedDocs,
        final int maxResultWindow,
        final float minScore
    ) throws IOException {
        final VectorScorer.Bulk bulkScorer = vectorScorer.bulk(matchedDocs);
        final DocAndFloatFeatureBuffer buffer = new DocAndFloatFeatureBuffer();
        final HitQueue queue = new HitQueue(maxResultWindow, true);
        ScoreDoc topDoc = queue.top();

        for (float maxBatchScore = bulkScorer.nextDocsAndScores(
            DocIdSetIterator.NO_MORE_DOCS,
            null,
            buffer
        ); buffer.size > 0; maxBatchScore = bulkScorer.nextDocsAndScores(DocIdSetIterator.NO_MORE_DOCS, null, buffer)) {
            if (maxBatchScore < minScore) {
                continue;
            }
            for (int i = 0; i < buffer.size; i++) {
                final float score = buffer.features[i];
                if (score >= minScore && score > topDoc.score) {
                    topDoc.score = score;
                    topDoc.doc = buffer.docs[i];
                    topDoc = queue.updateTop();
                }
            }
        }

        while (queue.size() > 0 && queue.top().score < 0) {
            queue.pop();
        }

        final ScoreDoc[] topScoreDocs = new ScoreDoc[queue.size()];
        for (int i = topScoreDocs.length - 1; i >= 0; i--) {
            topScoreDocs[i] = queue.pop();
        }
        return new TopDocs(new TotalHits(topScoreDocs.length, TotalHits.Relation.EQUAL_TO), topScoreDocs);
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
