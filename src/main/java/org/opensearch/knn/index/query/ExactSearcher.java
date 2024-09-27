/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import lombok.AllArgsConstructor;
import lombok.Builder;
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
import org.opensearch.knn.index.query.iterators.ByteVectorIdsKNNIterator;
import org.opensearch.knn.index.query.iterators.VectorIdsKNNIterator;
import org.opensearch.knn.index.query.iterators.KNNIterator;
import org.opensearch.knn.index.query.iterators.NestedByteVectorIdsKNNIterator;
import org.opensearch.knn.index.query.iterators.NestedVectorIdsKNNIterator;
import org.opensearch.knn.index.vectorvalues.KNNBinaryVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNFloatVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;
import org.opensearch.knn.indices.ModelDao;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

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
        KNNIterator iterator = getMatchedKNNIterator(leafReaderContext, exactSearcherContext);
        if (exactSearcherContext.getMatchedDocs().cardinality() <= exactSearcherContext.getK()) {
            return scoreAllDocs(iterator);
        }
        return searchTopK(iterator, exactSearcherContext.getK());
    }

    private Map<Integer, Float> scoreAllDocs(KNNIterator iterator) throws IOException {
        final Map<Integer, Float> docToScore = new HashMap<>();
        int docId;
        while ((docId = iterator.nextDoc()) != DocIdSetIterator.NO_MORE_DOCS) {
            docToScore.put(docId, iterator.score());
        }
        return docToScore;
    }

    private Map<Integer, Float> searchTopK(KNNIterator iterator, int k) throws IOException {
        // Creating min heap and init with MAX DocID and Score as -INF.
        final HitQueue queue = new HitQueue(k, true);
        ScoreDoc topDoc = queue.top();
        final Map<Integer, Float> docToScore = new HashMap<>();
        int docId;
        while ((docId = iterator.nextDoc()) != DocIdSetIterator.NO_MORE_DOCS) {
            if (iterator.score() > topDoc.score) {
                topDoc.score = iterator.score();
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

    private KNNIterator getMatchedKNNIterator(LeafReaderContext leafReaderContext, ExactSearcherContext exactSearcherContext)
        throws IOException {
        final KNNQuery knnQuery = exactSearcherContext.getKnnQuery();
        final BitSet matchedDocs = exactSearcherContext.getMatchedDocs();
        final SegmentReader reader = Lucene.segmentReader(leafReaderContext.reader());
        final FieldInfo fieldInfo = reader.getFieldInfos().fieldInfo(knnQuery.getField());
        final SpaceType spaceType = FieldInfoExtractor.getSpaceType(modelDao, fieldInfo);

        boolean isNestedRequired = exactSearcherContext.isParentHits() && knnQuery.getParentsFilter() != null;

        if (VectorDataType.BINARY == knnQuery.getVectorDataType() && isNestedRequired) {
            final KNNVectorValues<byte[]> vectorValues = KNNVectorValuesFactory.getVectorValues(fieldInfo, reader);
            return new NestedByteVectorIdsKNNIterator(
                matchedDocs,
                knnQuery.getByteQueryVector(),
                (KNNBinaryVectorValues) vectorValues,
                spaceType,
                knnQuery.getParentsFilter().getBitSet(leafReaderContext)
            );
        }

        if (VectorDataType.BINARY == knnQuery.getVectorDataType()) {
            final KNNVectorValues<byte[]> vectorValues = KNNVectorValuesFactory.getVectorValues(fieldInfo, reader);
            return new ByteVectorIdsKNNIterator(
                matchedDocs,
                knnQuery.getByteQueryVector(),
                (KNNBinaryVectorValues) vectorValues,
                spaceType
            );
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
