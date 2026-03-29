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

/**
 * Performs brute-force (exact) k-nearest-neighbor search over vector fields within individual
 * Lucene leaf segments. Unlike approximate search, every candidate vector is scored against the
 * query vector, guaranteeing perfect recall at the cost of higher latency.
 *
 * <p>Three search strategies are supported:
 * <ul>
 *   <li><b>Top-K</b> — returns the {@code k} highest-scoring documents using a min-heap.</li>
 *   <li><b>Score-all</b> — scores every matched document when the candidate set is smaller than
 *       or equal to {@code k}, avoiding unnecessary heap overhead.</li>
 *   <li><b>Radial (min-score)</b> — returns all documents whose similarity score meets or exceeds
 *       a minimum threshold derived from the query radius (FAISS engine only).</li>
 * </ul>
 *
 * <p>The searcher handles multiple {@link VectorDataType}s (float, byte, binary) and transparently
 * supports segment-level quantization when enabled, including Asymmetric Distance Computation (ADC).
 * For nested document structures, a parent {@link BitSet} is used to map child vectors back to
 * their parent documents.
 */
@Log4j2
@AllArgsConstructor
public class ExactSearcher {

    private final ModelDao modelDao;

    /**
     * Executes an exact search on a subset of documents within a single leaf segment.
     *
     * <p>The method selects the appropriate search strategy based on the provided context:
     * <ol>
     *   <li>If a {@code radius} is specified, delegates to {@link #doRadialSearch} which returns
     *       all documents meeting the minimum score threshold.</li>
     *   <li>Otherwise, delegates to {@link #exactNearestNeighborSearch} which selects between
     *       scoring all candidates or using a top-k heap based on the candidate set size.</li>
     * </ol>
     *
     * <p>For nested document structures (indicated by a non-null {@code parentsFilter} in the
     * context), the matched docs iterator is set to {@code null} because it has already been
     * consumed inside the nested vector scorer.
     *
     * @param leafReaderContext the {@link LeafReaderContext} providing access to the segment's reader
     * @param context           the {@link ExactSearcherContext} encapsulating query vector, k, radius,
     *                          matched document set, and other search parameters
     * @return {@link TopDocs} containing the scored results sorted by descending score;
     *         returns {@link TopDocsCollector#EMPTY_TOPDOCS} if no vector scorer can be created
     * @throws IOException if an I/O error occurs while reading vectors or computing scores
     */
    public TopDocs searchLeaf(final LeafReaderContext leafReaderContext, final ExactSearcherContext context) throws IOException {
        final SegmentReader reader = Lucene.segmentReader(leafReaderContext.reader());
        final FieldInfo fieldInfo = FieldInfoExtractor.getFieldInfo(reader, context.getField());
        if (fieldInfo == null) {
            log.debug("[KNN] FieldInfo is null for field [{}] in segment [{}]", context.getField(), reader.getSegmentName());
            return TopDocsCollector.EMPTY_TOPDOCS;
        }

        final VectorScorer vectorScorer = createVectorScorer(reader, fieldInfo, leafReaderContext, context);
        if (vectorScorer == null) {
            log.debug("[KNN] VectorScorer creation failed for field [{}] in segment [{}]", context.getField(), reader.getSegmentName());
            return TopDocsCollector.EMPTY_TOPDOCS;
        }

        // When nested, matchedDocsIterator is already consumed inside NestedBestChildVectorScorer,
        // so pass null to avoid double consumption of the same iterator.
        final boolean isNested = context.getParentsFilter() != null;
        final DocIdSetIterator matchedDocs = isNested ? null : context.getMatchedDocsIterator();

        if (context.getRadius() != null) {
            return doRadialSearch(fieldInfo, context, vectorScorer, matchedDocs);
        }
        return exactNearestNeighborSearch(context, vectorScorer, matchedDocs);
    }

    /**
     * Performs an exact nearest-neighbor search, choosing the optimal strategy based on the
     * candidate set size relative to {@code k}:
     * <ul>
     *   <li>If the number of matched documents is ≤ {@code k}, scores all candidates directly
     *       to avoid unnecessary heap overhead.</li>
     *   <li>Otherwise, uses a fixed-size min-heap to efficiently retain only the top-{@code k}
     *       results.</li>
     * </ul>
     *
     * @param context      the {@link ExactSearcherContext} containing {@code k} and matched doc metadata
     * @param vectorScorer the {@link VectorScorer} used to compute similarity scores
     * @param matchedDocs  a {@link DocIdSetIterator} over the candidate document set, or {@code null}
     *                     to score all documents available to the scorer
     * @return {@link TopDocs} containing the nearest-neighbor results sorted by descending score
     * @throws IOException if an I/O error occurs while reading vectors or computing scores
     */
    private TopDocs exactNearestNeighborSearch(
        final ExactSearcherContext context,
        final VectorScorer vectorScorer,
        final DocIdSetIterator matchedDocs
    ) throws IOException {
        if (context.getMatchedDocsIterator() != null && context.getNumberOfMatchedDocs() <= context.getK()) {
            return scoreAllDocs(vectorScorer, matchedDocs);
        }
        return searchTopK(vectorScorer, matchedDocs, context.getK());
    }

    /**
     * Performs a radial (distance-threshold) search by converting the query radius to a minimum
     * similarity score and returning all documents that meet or exceed that threshold.
     *
     * <p>Currently only the {@link KNNEngine#FAISS} engine supports radial search. The radius
     * value from the query is interpreted as a raw distance and is converted to a normalized
     * score via {@link KNNEngine#score(float, SpaceType)}, unless memory-optimized search is
     * enabled — in which case the radius is already expressed as a score.
     *
     * @param fieldInfo    the {@link FieldInfo} for the vector field (must not be {@code null})
     * @param context      the {@link ExactSearcherContext} containing the radius and other parameters
     * @param vectorScorer the {@link VectorScorer} used to compute similarity between the query
     *                     vector and each candidate document vector
     * @param matchedDocs  a {@link DocIdSetIterator} over the candidate document set, or {@code null}
     *                     if all documents in the segment should be considered
     * @return {@link TopDocs} containing all documents whose score meets the minimum threshold,
     *         up to {@code maxResultWindow} results, sorted by descending score
     * @throws IOException              if an I/O error occurs while reading vectors or computing scores
     * @throws IllegalArgumentException if the underlying engine is not FAISS
     */
    private TopDocs doRadialSearch(
        final FieldInfo fieldInfo,
        final ExactSearcherContext context,
        final VectorScorer vectorScorer,
        final DocIdSetIterator matchedDocs
    ) throws IOException {
        assert context.isMemoryOptimizedSearchEnabled != null;

        final KNNEngine engine = FieldInfoExtractor.extractKNNEngine(fieldInfo);
        if (KNNEngine.FAISS != engine) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "Engine [%s] does not support radial search", engine));
        }
        final SpaceType spaceType = FieldInfoExtractor.getSpaceType(modelDao, fieldInfo);
        final float minScore = context.isMemoryOptimizedSearchEnabled ? context.getRadius() : engine.score(context.getRadius(), spaceType);

        return searchWithMinScore(vectorScorer, matchedDocs, context.getMaxResultWindow(), minScore);
    }

    /**
     * Scores every candidate document against the query vector and returns all results sorted
     * by descending score. This method is used as an optimization when the total number of
     * matched documents is small enough (≤ k) that maintaining a bounded heap is unnecessary.
     *
     * @param vectorScorer the {@link VectorScorer} used to compute similarity scores
     * @param matchedDocs  a {@link DocIdSetIterator} over the candidate document set, or {@code null}
     *                     to score all documents available to the scorer
     * @return {@link TopDocs} containing all scored documents sorted by descending score
     * @throws IOException if an I/O error occurs while reading vectors or computing scores
     */
    private TopDocs scoreAllDocs(final VectorScorer vectorScorer, final DocIdSetIterator matchedDocs) throws IOException {
        final VectorScorer.Bulk bulkScorer = vectorScorer.bulk(matchedDocs);
        final DocAndFloatFeatureBuffer buffer = new DocAndFloatFeatureBuffer();
        final List<ScoreDoc> scoreDocList = new ArrayList<>();

        bulkScorer.nextDocsAndScores(DocIdSetIterator.NO_MORE_DOCS, null, buffer);
        while (buffer.size > 0) {
            for (int i = 0; i < buffer.size; i++) {
                scoreDocList.add(new ScoreDoc(buffer.docs[i], buffer.features[i]));
            }
            bulkScorer.nextDocsAndScores(DocIdSetIterator.NO_MORE_DOCS, null, buffer);
        }

        scoreDocList.sort(Comparator.comparing(scoreDoc -> scoreDoc.score, Comparator.reverseOrder()));
        return new TopDocs(new TotalHits(scoreDocList.size(), TotalHits.Relation.EQUAL_TO), scoreDocList.toArray(ScoreDoc[]::new));
    }

    /**
     * Finds the top-{@code k} highest-scoring documents using a fixed-size min-heap ({@link HitQueue}).
     * Documents are processed in batches via the bulk scorer; batches whose maximum score falls below
     * the current heap minimum are skipped entirely for efficiency.
     *
     * @param vectorScorer the {@link VectorScorer} used to compute similarity scores
     * @param matchedDocs  a {@link DocIdSetIterator} over the candidate document set, or {@code null}
     *                     to score all documents available to the scorer
     * @param k            the number of top results to return
     * @return {@link TopDocs} containing the {@code k} highest-scoring documents sorted by
     *         descending score; may contain fewer than {@code k} results if the candidate set
     *         is smaller or if sentinel entries are discarded
     * @throws IOException if an I/O error occurs while reading vectors or computing scores
     */
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

        return collectTopDocs(queue);
    }

    /**
     * Returns all documents whose similarity score meets or exceeds the specified minimum,
     * bounded by {@code maxResultWindow}. A fixed-size min-heap ({@link HitQueue}) is used to
     * retain the highest-scoring results. Batches whose maximum score falls below
     * {@code minScore} are skipped entirely for efficiency.
     *
     * @param vectorScorer   the {@link VectorScorer} used to compute similarity scores
     * @param matchedDocs    a {@link DocIdSetIterator} over the candidate document set, or {@code null}
     *                       to score all documents available to the scorer
     * @param maxResultWindow the maximum number of results to retain in the heap
     * @param minScore       the minimum similarity score a document must achieve to be included
     * @return {@link TopDocs} containing documents that meet the minimum score threshold,
     *         sorted by descending score, up to {@code maxResultWindow} results
     * @throws IOException if an I/O error occurs while reading vectors or computing scores
     */
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

        return collectTopDocs(queue);
    }

    /**
     * Collects scored results from the {@link HitQueue} into a {@link TopDocs}, discarding
     * sentinel entries (those with negative scores that were pre-populated when the queue was
     * created with {@code sentinelObjectsEnabled = true}). The remaining entries are returned
     * in descending score order.
     *
     * @param queue the {@link HitQueue} to collect from; will be empty after this call
     * @return {@link TopDocs} containing the valid results sorted by descending score
     */
    private static TopDocs collectTopDocs(final HitQueue queue) {
        while (queue.size() > 0 && queue.top().score < 0) {
            queue.pop();
        }
        final ScoreDoc[] topScoreDocs = new ScoreDoc[queue.size()];
        for (int i = topScoreDocs.length - 1; i >= 0; i--) {
            topScoreDocs[i] = queue.pop();
        }
        return new TopDocs(new TotalHits(topScoreDocs.length, TotalHits.Relation.EQUAL_TO), topScoreDocs);
    }

    /**
     * Creates the appropriate {@link VectorScorer} for the given field and segment, taking into
     * account the vector data type (float, byte, or binary), space type, quantization settings,
     * and nested document structure.
     *
     * <p>The scorer selection follows this precedence:
     * <ol>
     *   <li><b>Binary vectors</b> — uses the raw byte query vector directly.</li>
     *   <li><b>Byte vectors</b> — casts the float query vector to bytes before scoring.</li>
     *   <li><b>Float vectors (no quantization or rescore mode)</b> — uses the float query vector
     *       with full-precision document vectors.</li>
     *   <li><b>Float vectors with ADC quantization</b> — transforms the query vector via
     *       Asymmetric Distance Computation before scoring against quantized document vectors.</li>
     *   <li><b>Float vectors with other quantization</b> — quantizes the query vector and scores
     *       against quantized document vectors using Hamming distance.</li>
     * </ol>
     *
     * @param reader            the {@link SegmentReader} for the current segment
     * @param fieldInfo         the {@link FieldInfo} for the vector field (must not be {@code null})
     * @param leafReaderContext the {@link LeafReaderContext} used to resolve parent bit sets for
     *                          nested documents
     * @param context           the {@link ExactSearcherContext} containing the query vector and
     *                          search configuration
     * @return a configured {@link VectorScorer}, or {@code null} if scorer creation fails
     * @throws IOException if an I/O error occurs while reading vector values or quantization metadata
     */
    private VectorScorer createVectorScorer(
        final SegmentReader reader,
        final FieldInfo fieldInfo,
        final LeafReaderContext leafReaderContext,
        final ExactSearcherContext context
    ) throws IOException {
        final VectorDataType vectorDataType = FieldInfoExtractor.extractVectorDataType(fieldInfo);
        final SpaceType spaceType = FieldInfoExtractor.getSpaceType(modelDao, fieldInfo);
        final VectorScorerMode scorerMode = context.isUseQuantizedVectorsForSearch() ? VectorScorerMode.SCORE : VectorScorerMode.RESCORE;
        final boolean isNestedRequired = context.getParentsFilter() != null;
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
                fieldInfo,
                context.getMatchedDocsIterator(),
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
                fieldInfo,
                context.getMatchedDocsIterator(),
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
                context.getMatchedDocsIterator(),
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
                context.getMatchedDocsIterator(),
                parentBitSet
            );
        }

        final byte[] quantizedQueryVector = SegmentLevelQuantizationUtil.quantizeVector(context.getFloatQueryVector(), quantizationInfo);
        return VectorScorers.createScorer(
            quantizedIteratorValues,
            quantizedQueryVector,
            scorerMode,
            SpaceType.HAMMING,
            fieldInfo,
            context.getMatchedDocsIterator(),
            parentBitSet
        );
    }

    /**
     * Immutable context object that encapsulates all parameters required to execute an exact search.
     * Constructed via the Lombok {@link Builder} pattern to avoid constructor parameter explosion.
     */
    @Value
    @Builder
    public static class ExactSearcherContext {
        /**
         * When {@code true}, the searcher uses quantized (compressed) vectors for scoring.
         * When {@code false}, full-precision vectors are used instead — this is necessary
         * during re-scoring to preserve accuracy.
         */
        boolean useQuantizedVectorsForSearch;

        /** The number of nearest neighbors to return. */
        int k;

        /**
         * The search radius for radial (distance-threshold) queries. When non-null, the search
         * returns all documents within this distance rather than a fixed top-k. The value is
         * interpreted as a raw distance (converted to a score internally) unless memory-optimized
         * search is enabled, in which case it is already a score.
         */
        Float radius;

        /**
         * An optional iterator over the pre-filtered candidate document set. When {@code null},
         * all documents in the segment are considered. For nested queries, this iterator may
         * already be consumed by the nested vector scorer.
         */
        @Nullable
        DocIdSetIterator matchedDocsIterator;

        /** The total number of documents matched by the pre-filter, used to select the search strategy. */
        long numberOfMatchedDocs;

        /**
         * A {@link BitSetProducer} that identifies parent documents in a nested document structure.
         * When non-null, indicates that the search operates over nested (child) vectors and the
         * results must be mapped back to parent document IDs. In filtered nested search scenarios,
         * the matched docs contain parent IDs and a {@code NestedVectorIdsExactKNNIterator} is used.
         */
        BitSetProducer parentsFilter;

        /** The query vector in float representation, used for float and byte vector data types. */
        float[] floatQueryVector;

        /** The query vector in byte representation, used for binary vector data types. */
        byte[] byteQueryVector;

        /** The name of the k-NN vector field being searched. */
        String field;

        /**
         * The maximum number of results to retain during radial search. Acts as an upper bound
         * on the heap size in {@link #searchWithMinScore} to prevent unbounded memory usage.
         */
        Integer maxResultWindow;

        /** The Lucene {@link VectorSimilarityFunction} associated with the vector field. */
        VectorSimilarityFunction similarityFunction;

        /**
         * When {@code true}, indicates that memory-optimized search is enabled, meaning the
         * radius value is already expressed as a normalized score and does not require conversion
         * from a raw distance.
         */
        Boolean isMemoryOptimizedSearchEnabled;
    }
}
