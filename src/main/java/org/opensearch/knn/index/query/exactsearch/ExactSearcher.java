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
import org.apache.lucene.search.ConjunctionUtils;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.HitQueue;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopDocsCollector;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.search.join.BitSetProducer;
import org.opensearch.common.Nullable;
import org.opensearch.common.lucene.Lucene;
import org.opensearch.knn.common.FieldInfoExtractor;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.query.SegmentLevelQuantizationInfo;
import org.opensearch.knn.index.query.SegmentLevelQuantizationUtil;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.vectorvalues.KNNBinaryVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNByteVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNFloatVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;
import org.opensearch.knn.indices.ModelDao;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
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
        final ExactKNNIterator iterator = getKNNIterator(leafReaderContext, context);
        // because of any reason if we are not able to get ExactKNNIterator, return empty top docss
        if (iterator == null) {
            return TopDocsCollector.EMPTY_TOPDOCS;
        }
        if (context.getRadius() != null) {
            return doRadialSearch(leafReaderContext, context, iterator);
        }
        if (context.getMatchedDocsIterator() != null && context.numberOfMatchedDocs <= context.getK()) {
            return scoreAllDocs(iterator);
        }
        return searchTopCandidates(iterator, context.getK(), Predicates.alwaysTrue());
    }

    /**
     * Perform radial search by comparing scores with min score. Currently, FAISS from native engine supports radial search.
     * Hence, we assume that Radius from knnQuery is always distance, and we convert it to score since we do exact search uses scores
     * to filter out the documents that does not have given min score.
     * @param leafReaderContext {@link LeafReaderContext}
     * @param context {@link ExactSearcherContext}
     * @param iterator {@link ExactKNNIterator}
     * @return TopDocs containing the results of the search
     * @throws IOException exception raised by iterator during traversal
     */
    private TopDocs doRadialSearch(LeafReaderContext leafReaderContext, ExactSearcherContext context, ExactKNNIterator iterator)
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

        return filterDocsByMinScore(context, iterator, minScore);
    }

    private TopDocs scoreAllDocs(ExactKNNIterator iterator) throws IOException {
        final List<ScoreDoc> scoreDocList = new ArrayList<>();
        int docId;
        while ((docId = iterator.nextDoc()) != DocIdSetIterator.NO_MORE_DOCS) {
            scoreDocList.add(new ScoreDoc(docId, iterator.score()));
        }
        scoreDocList.sort(Comparator.comparing(scoreDoc -> scoreDoc.score, Comparator.reverseOrder()));
        return new TopDocs(new TotalHits(scoreDocList.size(), TotalHits.Relation.EQUAL_TO), scoreDocList.toArray(ScoreDoc[]::new));
    }

    private TopDocs searchTopCandidates(ExactKNNIterator iterator, int limit, @NonNull Predicate<Float> filterScore) throws IOException {
        // Creating min heap and init with MAX DocID and Score as -INF.
        final HitQueue queue = new HitQueue(limit, true);
        ScoreDoc topDoc = queue.top();
        final Map<Integer, Float> docToScore = new HashMap<>();
        int docId;
        while ((docId = iterator.nextDoc()) != DocIdSetIterator.NO_MORE_DOCS) {
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

    private TopDocs filterDocsByMinScore(ExactSearcherContext context, ExactKNNIterator iterator, float minScore) throws IOException {
        int maxResultWindow = context.getMaxResultWindow();
        Predicate<Float> scoreGreaterThanOrEqualToMinScore = score -> score >= minScore;
        return searchTopCandidates(iterator, maxResultWindow, scoreGreaterThanOrEqualToMinScore);
    }

    private ExactKNNIterator getKNNIterator(LeafReaderContext leafReaderContext, ExactSearcherContext exactSearcherContext)
        throws IOException {
        final SegmentReader reader = Lucene.segmentReader(leafReaderContext.reader());
        final FieldInfo fieldInfo = FieldInfoExtractor.getFieldInfo(reader, exactSearcherContext.getField());
        if (fieldInfo == null) {
            log.debug(
                "[KNN] Cannot get ExactKNNIterator as Field info not found for {}:{}",
                exactSearcherContext.getField(),
                reader.getSegmentName()
            );
            return null;
        }
        final VectorDataType vectorDataType = FieldInfoExtractor.extractVectorDataType(fieldInfo);
        final SpaceType spaceType = FieldInfoExtractor.getSpaceType(modelDao, fieldInfo);
        boolean isNestedRequired = exactSearcherContext.getParentsFilter() != null;

        // We need to create a new VectorValues instances as the new one will be used to iterate over the docIds in
        // conjunction with Matched Docs.
        final DocIdSetIterator matchedDocs = getMatchedDocsIterator(
            exactSearcherContext.getMatchedDocsIterator(),
            KNNVectorValuesFactory.getVectorValues(fieldInfo, reader).getVectorValuesIterator().getDocIdSetIterator()
        );
        if (VectorDataType.BINARY == vectorDataType) {
            final KNNVectorValues<byte[]> vectorValues = KNNVectorValuesFactory.getVectorValues(fieldInfo, reader);
            if (isNestedRequired) {
                return new NestedBinaryVectorIdsExactKNNIterator(
                    matchedDocs,
                    exactSearcherContext.getByteQueryVector(),
                    (KNNBinaryVectorValues) vectorValues,
                    spaceType,
                    exactSearcherContext.getParentsFilter().getBitSet(leafReaderContext)
                );
            }
            return new BinaryVectorIdsExactKNNIterator(
                matchedDocs,
                exactSearcherContext.getByteQueryVector(),
                (KNNBinaryVectorValues) vectorValues,
                spaceType
            );
        }

        if (VectorDataType.BYTE == vectorDataType) {
            final KNNVectorValues<byte[]> vectorValues = KNNVectorValuesFactory.getVectorValues(fieldInfo, reader);
            if (isNestedRequired) {
                return new NestedByteVectorIdsExactKNNIterator(
                    matchedDocs,
                    exactSearcherContext.getFloatQueryVector(),
                    (KNNByteVectorValues) vectorValues,
                    spaceType,
                    exactSearcherContext.getParentsFilter().getBitSet(leafReaderContext)
                );
            }
            return new ByteVectorIdsExactKNNIterator(
                matchedDocs,
                exactSearcherContext.getFloatQueryVector(),
                (KNNByteVectorValues) vectorValues,
                spaceType
            );
        }
        // Build Segment Level Quantization info.
        final SegmentLevelQuantizationInfo segmentLevelQuantizationInfo = SegmentLevelQuantizationInfo.build(
            reader,
            fieldInfo,
            exactSearcherContext.getField()
        );
        // For FP32 vectors, there are two execution paths:
        // 1. Full precision path: Used during rescoring or when quantization is not available.
        // Loads original float32 vectors and performs exact search using the configured distance metric (L2, Cosine, etc.).
        // 2. Quantized path: Used during approximate search when quantization is enabled.
        // Loads quantized byte vectors from segment and performs search using either:
        // a) ADC (Asymmetric Distance Computation): Transforms query vector and compares against quantized doc vectors
        // b) Symmetric quantization: Quantizes query vector and uses Hamming distance for comparison
        if (segmentLevelQuantizationInfo == null || !exactSearcherContext.isUseQuantizedVectorsForSearch()) {
            final KNNVectorValues<float[]> vectorValues = KNNVectorValuesFactory.getVectorValues(fieldInfo, reader);
            if (isNestedRequired) {
                return new NestedVectorIdsExactKNNIterator(
                    matchedDocs,
                    exactSearcherContext.getFloatQueryVector(),
                    (KNNFloatVectorValues) vectorValues,
                    spaceType,
                    exactSearcherContext.getParentsFilter().getBitSet(leafReaderContext)
                );
            }
            return new VectorIdsExactKNNIterator(
                matchedDocs,
                exactSearcherContext.getFloatQueryVector(),
                (KNNFloatVectorValues) vectorValues,
                spaceType
            );
        }
        final KNNVectorValues<byte[]> vectorValues = KNNVectorValuesFactory.getVectorValues(fieldInfo, reader, true);
        // For ADC, we will transform float vector -> ADC's float vector
        if (SegmentLevelQuantizationUtil.isAdcEnabled(segmentLevelQuantizationInfo)) {
            SegmentLevelQuantizationUtil.transformVectorWithADC(
                exactSearcherContext.getFloatQueryVector(),
                segmentLevelQuantizationInfo,
                spaceType
            );
            if (isNestedRequired) {
                return new NestedBinaryVectorIdsExactKNNIterator(
                    matchedDocs,
                    exactSearcherContext.getFloatQueryVector(),
                    (KNNBinaryVectorValues) vectorValues,
                    spaceType,
                    exactSearcherContext.getParentsFilter().getBitSet(leafReaderContext)
                );
            }
            return new BinaryVectorIdsExactKNNIterator(
                matchedDocs,
                exactSearcherContext.getFloatQueryVector(),
                (KNNBinaryVectorValues) vectorValues,
                spaceType
            );
        }
        final byte[] quantizedQueryVector = SegmentLevelQuantizationUtil.quantizeVector(
            exactSearcherContext.getFloatQueryVector(),
            segmentLevelQuantizationInfo
        );
        // Quantized search path: retrieve quantized byte vectors from reader as KNNBinaryVectorValues and perform exact search
        // using Hamming distance.
        if (isNestedRequired) {
            return new NestedBinaryVectorIdsExactKNNIterator(
                matchedDocs,
                quantizedQueryVector,
                (KNNBinaryVectorValues) vectorValues,
                SpaceType.HAMMING,
                exactSearcherContext.getParentsFilter().getBitSet(leafReaderContext)
            );
        }
        return new BinaryVectorIdsExactKNNIterator(
            matchedDocs,
            quantizedQueryVector,
            (KNNBinaryVectorValues) vectorValues,
            SpaceType.HAMMING
        );
    }

    /**
    * Creates a {@link DocIdSetIterator} which is an intersection of the iterators passed as arguments.
    * This is used to get the intersection of the matched docs and the vector values docIds.
    *
    * @param originalMatchedDocsDISI A {@link DocIdSetIterator} on which exact search needs to be performed
    * @param vectorValuesDISI A {@link DocIdSetIterator} which contains the docIds which has vector on it.
    * @return DocIdSetIterator A intersection of the iterators
    */
    private DocIdSetIterator getMatchedDocsIterator(
        @Nullable final DocIdSetIterator originalMatchedDocsDISI,
        final DocIdSetIterator vectorValuesDISI
    ) {
        if (originalMatchedDocsDISI == null) {
            return null;
        }
        final List<DocIdSetIterator> disiList = new ArrayList<>();
        disiList.add(originalMatchedDocsDISI);
        disiList.add(vectorValuesDISI);
        return ConjunctionUtils.intersectIterators(disiList);
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
