/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import com.google.common.base.Predicates;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.NonNull;
import lombok.Value;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.HitQueue;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.util.BitSet;
import org.opensearch.common.lucene.Lucene;
import org.opensearch.knn.common.FieldInfoExtractor;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.query.iterators.BinaryVectorIdsKNNIterator;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.query.iterators.ByteVectorIdsKNNIterator;
import org.opensearch.knn.index.query.iterators.NestedBinaryVectorIdsKNNIterator;
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

import java.io.IOException;
import java.util.HashMap;
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
     * @param exactSearcherContext {@link ExactSearcherContext}
     * @return Map of re-scored results
     * @throws IOException exception during execution of exact search
     */
    public Map<Integer, Float> searchLeaf(final LeafReaderContext leafReaderContext, final ExactSearcherContext exactSearcherContext)
        throws IOException {
        KNNIterator iterator = getKNNIterator(leafReaderContext, exactSearcherContext);
        if (exactSearcherContext.getKnnQuery().getRadius() != null) {
            return doRadialSearch(leafReaderContext, exactSearcherContext, iterator);
        }
        if (exactSearcherContext.getMatchedDocs() != null
            && exactSearcherContext.getMatchedDocs().cardinality() <= exactSearcherContext.getK()) {
            return scoreAllDocs(iterator);
        }
        return searchTopCandidates(iterator, exactSearcherContext.getK(), Predicates.alwaysTrue());
    }

    /**
     * Perform radial search by comparing scores with min score. Currently, FAISS from native engine supports radial search.
     * Hence, we assume that Radius from knnQuery is always distance, and we convert it to score since we do exact search uses scores
     * to filter out the documents that does not have given min score.
     * @param leafReaderContext
     * @param exactSearcherContext
     * @param iterator {@link KNNIterator}
     * @return Map of docId and score
     * @throws IOException exception raised by iterator during traversal
     */
    private Map<Integer, Float> doRadialSearch(
        LeafReaderContext leafReaderContext,
        ExactSearcherContext exactSearcherContext,
        KNNIterator iterator
    ) throws IOException {
        final SegmentReader reader = Lucene.segmentReader(leafReaderContext.reader());
        final KNNQuery knnQuery = exactSearcherContext.getKnnQuery();
        final FieldInfo fieldInfo = reader.getFieldInfos().fieldInfo(knnQuery.getField());
        final KNNEngine engine = FieldInfoExtractor.extractKNNEngine(fieldInfo);
        if (KNNEngine.FAISS != engine) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "Engine [%s] does not support radial search", engine));
        }
        final SpaceType spaceType = FieldInfoExtractor.getSpaceType(modelDao, fieldInfo);
        final float minScore = spaceType.scoreTranslation(knnQuery.getRadius());
        return filterDocsByMinScore(exactSearcherContext, iterator, minScore);
    }

    private Map<Integer, Float> scoreAllDocs(KNNIterator iterator) throws IOException {
        final Map<Integer, Float> docToScore = new HashMap<>();
        int docId;
        while ((docId = iterator.nextDoc()) != DocIdSetIterator.NO_MORE_DOCS) {
            docToScore.put(docId, iterator.score());
        }
        return docToScore;
    }

    private Map<Integer, Float> searchTopCandidates(KNNIterator iterator, int limit, @NonNull Predicate<Float> filterScore)
        throws IOException {
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
        // If filterIds < k, the some values in heap can have a negative score.
        while (queue.size() > 0 && queue.top().score < 0) {
            queue.pop();
        }

        while (queue.size() > 0) {
            final ScoreDoc doc = queue.pop();
            docToScore.put(doc.doc, doc.score);
        }
        return docToScore;
    }

    private Map<Integer, Float> filterDocsByMinScore(ExactSearcherContext context, KNNIterator iterator, float minScore)
        throws IOException {
        int maxResultWindow = context.getKnnQuery().getContext().getMaxResultWindow();
        Predicate<Float> scoreGreaterThanOrEqualToMinScore = score -> score >= minScore;
        return searchTopCandidates(iterator, maxResultWindow, scoreGreaterThanOrEqualToMinScore);
    }

    private KNNIterator getKNNIterator(LeafReaderContext leafReaderContext, ExactSearcherContext exactSearcherContext) throws IOException {
        final KNNQuery knnQuery = exactSearcherContext.getKnnQuery();
        final BitSet matchedDocs = exactSearcherContext.getMatchedDocs();
        final SegmentReader reader = Lucene.segmentReader(leafReaderContext.reader());
        final FieldInfo fieldInfo = reader.getFieldInfos().fieldInfo(knnQuery.getField());
        final SpaceType spaceType = FieldInfoExtractor.getSpaceType(modelDao, fieldInfo);

        boolean isNestedRequired = exactSearcherContext.isParentHits() && knnQuery.getParentsFilter() != null;

        if (VectorDataType.BINARY == knnQuery.getVectorDataType()) {
            final KNNVectorValues<byte[]> vectorValues = KNNVectorValuesFactory.getVectorValues(fieldInfo, reader);
            if (isNestedRequired) {
                return new NestedBinaryVectorIdsKNNIterator(
                    matchedDocs,
                    knnQuery.getByteQueryVector(),
                    (KNNBinaryVectorValues) vectorValues,
                    spaceType,
                    knnQuery.getParentsFilter().getBitSet(leafReaderContext)
                );
            }
            return new BinaryVectorIdsKNNIterator(
                matchedDocs,
                knnQuery.getByteQueryVector(),
                (KNNBinaryVectorValues) vectorValues,
                spaceType
            );
        }

        if (VectorDataType.BYTE == knnQuery.getVectorDataType()) {
            final KNNVectorValues<byte[]> vectorValues = KNNVectorValuesFactory.getVectorValues(fieldInfo, reader);
            if (isNestedRequired) {
                return new NestedByteVectorIdsKNNIterator(
                    matchedDocs,
                    knnQuery.getQueryVector(),
                    (KNNByteVectorValues) vectorValues,
                    spaceType,
                    knnQuery.getParentsFilter().getBitSet(leafReaderContext)
                );
            }
            return new ByteVectorIdsKNNIterator(matchedDocs, knnQuery.getQueryVector(), (KNNByteVectorValues) vectorValues, spaceType);
        }
        final byte[] quantizedQueryVector;
        final SegmentLevelQuantizationInfo segmentLevelQuantizationInfo;
        if (exactSearcherContext.isUseQuantizedVectorsForSearch()) {
            // Build Segment Level Quantization info.
            segmentLevelQuantizationInfo = SegmentLevelQuantizationInfo.build(reader, fieldInfo, knnQuery.getField());
            // Quantize the Query Vector Once.
            quantizedQueryVector = SegmentLevelQuantizationUtil.quantizeVector(knnQuery.getQueryVector(), segmentLevelQuantizationInfo);
        } else {
            segmentLevelQuantizationInfo = null;
            quantizedQueryVector = null;
        }

        final KNNVectorValues<float[]> vectorValues = KNNVectorValuesFactory.getVectorValues(fieldInfo, reader);
        if (isNestedRequired) {
            return new NestedVectorIdsKNNIterator(
                matchedDocs,
                knnQuery.getQueryVector(),
                (KNNFloatVectorValues) vectorValues,
                spaceType,
                knnQuery.getParentsFilter().getBitSet(leafReaderContext),
                quantizedQueryVector,
                segmentLevelQuantizationInfo
            );
        }
        return new VectorIdsKNNIterator(
            matchedDocs,
            knnQuery.getQueryVector(),
            (KNNFloatVectorValues) vectorValues,
            spaceType,
            quantizedQueryVector,
            segmentLevelQuantizationInfo
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
        BitSet matchedDocs;
        KNNQuery knnQuery;
        /**
         * whether the matchedDocs contains parent ids or child ids. This is relevant in the case of
         * filtered nested search where the matchedDocs contain the parent ids and {@link NestedVectorIdsKNNIterator}
         * needs to be used.
         */
        boolean isParentHits;
    }
}
