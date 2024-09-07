/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import com.google.common.annotations.VisibleForTesting;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.Explanation;
import org.apache.lucene.search.FilteredDocIdSetIterator;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.Weight;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.FilterDirectory;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.BitSetIterator;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;
import org.opensearch.common.io.PathUtils;
import org.opensearch.common.lucene.Lucene;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.KNN990Codec.QuantizationConfigKNNCollector;
import org.opensearch.knn.index.memory.NativeMemoryAllocation;
import org.opensearch.knn.index.memory.NativeMemoryCacheManager;
import org.opensearch.knn.index.memory.NativeMemoryEntryContext;
import org.opensearch.knn.index.memory.NativeMemoryLoadStrategy;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.quantizationservice.QuantizationService;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.indices.ModelUtil;
import org.opensearch.knn.jni.JNIService;
import org.opensearch.knn.plugin.stats.KNNCounter;
import org.opensearch.knn.quantization.models.quantizationOutput.QuantizationOutput;
import org.opensearch.knn.quantization.models.quantizationParams.QuantizationParams;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.stream.Collectors;

import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.MODEL_ID;
import static org.opensearch.knn.common.KNNConstants.SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;
import static org.opensearch.knn.index.util.IndexUtil.getParametersAtLoading;
import static org.opensearch.knn.plugin.stats.KNNCounter.GRAPH_QUERY_ERRORS;

/**
 * Calculate query weights and build query scorers.
 */
@Log4j2
public class KNNWeight extends Weight {
    private static ModelDao modelDao;

    private final KNNQuery knnQuery;
    private final float boost;

    private final NativeMemoryCacheManager nativeMemoryCacheManager;
    private final Weight filterWeight;
    private final ExactSearcher exactSearcher;

    private static ExactSearcher DEFAULT_EXACT_SEARCHER;
    private final QuantizationService quantizationService;

    public KNNWeight(KNNQuery query, float boost) {
        super(query);
        this.knnQuery = query;
        this.boost = boost;
        this.nativeMemoryCacheManager = NativeMemoryCacheManager.getInstance();
        this.filterWeight = null;
        this.exactSearcher = DEFAULT_EXACT_SEARCHER;
        this.quantizationService = QuantizationService.getInstance();
    }

    public KNNWeight(KNNQuery query, float boost, Weight filterWeight) {
        super(query);
        this.knnQuery = query;
        this.boost = boost;
        this.nativeMemoryCacheManager = NativeMemoryCacheManager.getInstance();
        this.filterWeight = filterWeight;
        this.exactSearcher = DEFAULT_EXACT_SEARCHER;
        this.quantizationService = QuantizationService.getInstance();
    }

    public static void initialize(ModelDao modelDao) {
        KNNWeight.modelDao = modelDao;
        KNNWeight.DEFAULT_EXACT_SEARCHER = new ExactSearcher(modelDao);
    }

    @Override
    public Explanation explain(LeafReaderContext context, int doc) {
        return Explanation.match(1.0f, "No Explanation");
    }

    @Override
    public Scorer scorer(LeafReaderContext context) throws IOException {
        final Map<Integer, Float> docIdToScoreMap = searchLeaf(context, knnQuery.getK());
        if (docIdToScoreMap.isEmpty()) {
            return KNNScorer.emptyScorer(this);
        }
        final int maxDoc = Collections.max(docIdToScoreMap.keySet()) + 1;
        return new KNNScorer(this, ResultUtil.resultMapToDocIds(docIdToScoreMap, maxDoc), docIdToScoreMap, boost);
    }

    /**
     * Executes k nearest neighbor search for a segment to get the top K results
     * This is made public purely to be able to be reused in {@link org.opensearch.knn.index.query.nativelib.NativeEngineKnnVectorQuery}
     *
     * @param context LeafReaderContext
     * @param k Number of results to return
     * @return A Map of docId to scores for top k results
     */
    public Map<Integer, Float> searchLeaf(LeafReaderContext context, int k) throws IOException {
        final BitSet filterBitSet = getFilteredDocsBitSet(context);
        int cardinality = filterBitSet.cardinality();
        // We don't need to go to JNI layer if no documents are found which satisfy the filters
        // We should give this condition a deeper look that where it should be placed. For now I feel this is a good
        // place,
        if (filterWeight != null && cardinality == 0) {
            return Collections.emptyMap();
        }

        /*
         * The idea for this optimization is to get K results, we need to atleast look at K vectors in the HNSW graph
         * . Hence, if filtered results are less than K and filter query is present we should shift to exact search.
         * This improves the recall.
         */
        Map<Integer, Float> docIdsToScoreMap;
        if (filterWeight != null && canDoExactSearch(cardinality)) {
            docIdsToScoreMap = exactSearch(context, filterBitSet, true, k);
        } else {
            docIdsToScoreMap = doANNSearch(context, filterBitSet, cardinality, k);
            if (docIdsToScoreMap == null) {
                return Collections.emptyMap();
            }
            if (canDoExactSearchAfterANNSearch(cardinality, docIdsToScoreMap.size())) {
                log.debug(
                    "Doing ExactSearch after doing ANNSearch as the number of documents returned are less than "
                        + "K, even when we have more than K filtered Ids. K: {}, ANNResults: {}, filteredIdCount: {}",
                    k,
                    docIdsToScoreMap.size(),
                    cardinality
                );
                docIdsToScoreMap = exactSearch(context, filterBitSet, true, k);
            }
        }
        if (docIdsToScoreMap.isEmpty()) {
            return Collections.emptyMap();
        }
        return docIdsToScoreMap;
    }

    private BitSet getFilteredDocsBitSet(final LeafReaderContext ctx) throws IOException {
        if (this.filterWeight == null) {
            return new FixedBitSet(0);
        }

        final Bits liveDocs = ctx.reader().getLiveDocs();
        final int maxDoc = ctx.reader().maxDoc();

        final Scorer scorer = filterWeight.scorer(ctx);
        if (scorer == null) {
            return new FixedBitSet(0);
        }

        return createBitSet(scorer.iterator(), liveDocs, maxDoc);
    }

    private BitSet createBitSet(final DocIdSetIterator filteredDocIdsIterator, final Bits liveDocs, int maxDoc) throws IOException {
        if (liveDocs == null && filteredDocIdsIterator instanceof BitSetIterator) {
            // If we already have a BitSet and no deletions, reuse the BitSet
            return ((BitSetIterator) filteredDocIdsIterator).getBitSet();
        }
        // Create a new BitSet from matching and live docs
        FilteredDocIdSetIterator filterIterator = new FilteredDocIdSetIterator(filteredDocIdsIterator) {
            @Override
            protected boolean match(int doc) {
                return liveDocs == null || liveDocs.get(doc);
            }
        };
        return BitSet.of(filterIterator, maxDoc);
    }

    private int[] getParentIdsArray(final LeafReaderContext context) throws IOException {
        if (knnQuery.getParentsFilter() == null) {
            return null;
        }
        return bitSetToIntArray(knnQuery.getParentsFilter().getBitSet(context));
    }

    private int[] bitSetToIntArray(final BitSet bitSet) {
        final int cardinality = bitSet.cardinality();
        final int[] intArray = new int[cardinality];
        final BitSetIterator bitSetIterator = new BitSetIterator(bitSet, cardinality);
        int index = 0;
        int docId = bitSetIterator.nextDoc();
        while (docId != DocIdSetIterator.NO_MORE_DOCS) {
            assert index < intArray.length;
            intArray[index++] = docId;
            docId = bitSetIterator.nextDoc();
        }
        return intArray;
    }

    private Map<Integer, Float> doANNSearch(
        final LeafReaderContext context,
        final BitSet filterIdsBitSet,
        final int cardinality,
        final int k
    ) throws IOException {
        final SegmentReader reader = Lucene.segmentReader(context.reader());
        String directory = ((FSDirectory) FilterDirectory.unwrap(reader.directory())).getDirectory().toString();

        FieldInfo fieldInfo = reader.getFieldInfos().fieldInfo(knnQuery.getField());

        if (fieldInfo == null) {
            log.debug("[KNN] Field info not found for {}:{}", knnQuery.getField(), reader.getSegmentName());
            return null;
        }

        KNNEngine knnEngine;
        SpaceType spaceType;
        VectorDataType vectorDataType;

        // Check if a modelId exists. If so, the space type and engine will need to be picked up from the model's
        // metadata.
        String modelId = fieldInfo.getAttribute(MODEL_ID);
        if (modelId != null) {
            ModelMetadata modelMetadata = modelDao.getMetadata(modelId);
            if (!ModelUtil.isModelCreated(modelMetadata)) {
                throw new RuntimeException("Model \"" + modelId + "\" is not created.");
            }

            knnEngine = modelMetadata.getKnnEngine();
            spaceType = modelMetadata.getSpaceType();
            vectorDataType = modelMetadata.getVectorDataType();
        } else {
            String engineName = fieldInfo.attributes().getOrDefault(KNN_ENGINE, KNNEngine.NMSLIB.getName());
            knnEngine = KNNEngine.getEngine(engineName);
            String spaceTypeName = fieldInfo.attributes().getOrDefault(SPACE_TYPE, SpaceType.L2.getValue());
            spaceType = SpaceType.getSpace(spaceTypeName);
            vectorDataType = VectorDataType.get(
                fieldInfo.attributes().getOrDefault(VECTOR_DATA_TYPE_FIELD, VectorDataType.FLOAT.getValue())
            );
        }

        QuantizationParams quantizationParams = quantizationService.getQuantizationParams(fieldInfo);

        // TODO: Change type of vector once more quantization methods are supported
        byte[] quantizedVector = getQuantizedVector(quantizationParams, reader, fieldInfo);

        List<String> engineFiles = getEngineFiles(reader, knnEngine.getExtension());
        if (engineFiles.isEmpty()) {
            log.debug("[KNN] No engine index found for field {} for segment {}", knnQuery.getField(), reader.getSegmentName());
            return null;
        }

        Path indexPath = PathUtils.get(directory, engineFiles.get(0));
        final KNNQueryResult[] results;
        KNNCounter.GRAPH_QUERY_REQUESTS.increment();

        // We need to first get index allocation
        NativeMemoryAllocation indexAllocation;
        try {
            indexAllocation = nativeMemoryCacheManager.get(
                new NativeMemoryEntryContext.IndexEntryContext(
                    indexPath.toString(),
                    NativeMemoryLoadStrategy.IndexLoadStrategy.getInstance(),
                    getParametersAtLoading(
                        spaceType,
                        knnEngine,
                        knnQuery.getIndexName(),
                        // TODO: In the future, more vector data types will be supported with quantization
                        quantizationParams == null ? vectorDataType : VectorDataType.BINARY
                    ),
                    knnQuery.getIndexName(),
                    modelId
                ),
                true
            );
        } catch (ExecutionException e) {
            GRAPH_QUERY_ERRORS.increment();
            throw new RuntimeException(e);
        }

        // From cardinality select different filterIds type
        FilterIdsSelector filterIdsSelector = FilterIdsSelector.getFilterIdSelector(filterIdsBitSet, cardinality);
        long[] filterIds = filterIdsSelector.getFilterIds();
        FilterIdsSelector.FilterIdsSelectorType filterType = filterIdsSelector.getFilterType();
        // Now that we have the allocation, we need to readLock it
        indexAllocation.readLock();
        try {
            if (indexAllocation.isClosed()) {
                throw new RuntimeException("Index has already been closed");
            }
            int[] parentIds = getParentIdsArray(context);
            if (k > 0) {
                if (knnQuery.getVectorDataType() == VectorDataType.BINARY
                    || quantizationParams != null && quantizationService.getVectorDataTypeForTransfer(fieldInfo) == VectorDataType.BINARY) {
                    results = JNIService.queryBinaryIndex(
                        indexAllocation.getMemoryAddress(),
                        // TODO: In the future, quantizedVector can have other data types than byte
                        quantizationParams == null ? knnQuery.getByteQueryVector() : quantizedVector,
                        k,
                        knnQuery.getMethodParameters(),
                        knnEngine,
                        filterIds,
                        filterType.getValue(),
                        parentIds
                    );
                } else {
                    results = JNIService.queryIndex(
                        indexAllocation.getMemoryAddress(),
                        knnQuery.getQueryVector(),
                        k,
                        knnQuery.getMethodParameters(),
                        knnEngine,
                        filterIds,
                        filterType.getValue(),
                        parentIds
                    );
                }
            } else {
                results = JNIService.radiusQueryIndex(
                    indexAllocation.getMemoryAddress(),
                    knnQuery.getQueryVector(),
                    knnQuery.getRadius(),
                    knnQuery.getMethodParameters(),
                    knnEngine,
                    knnQuery.getContext().getMaxResultWindow(),
                    filterIds,
                    filterType.getValue(),
                    parentIds
                );
            }
        } catch (Exception e) {
            GRAPH_QUERY_ERRORS.increment();
            throw new RuntimeException(e);
        } finally {
            indexAllocation.readUnlock();
        }

        /*
         * Scores represent the distance of the documents with respect to given query vector.
         * Lesser the score, the closer the document is to the query vector.
         * Since by default results are retrieved in the descending order of scores, to get the nearest
         * neighbors we are inverting the scores.
         */
        if (results.length == 0) {
            log.debug("[KNN] Query yielded 0 results");
            return null;
        }

        return Arrays.stream(results)
            .collect(Collectors.toMap(KNNQueryResult::getId, result -> knnEngine.score(result.getScore(), spaceType)));
    }

    @VisibleForTesting
    List<String> getEngineFiles(SegmentReader reader, String extension) throws IOException {
        /*
         * In case of compound file, extension would be <engine-extension> + c otherwise <engine-extension>
         */
        String engineExtension = reader.getSegmentInfo().info.getUseCompoundFile()
            ? extension + KNNConstants.COMPOUND_EXTENSION
            : extension;
        String engineSuffix = knnQuery.getField() + engineExtension;
        String underLineEngineSuffix = "_" + engineSuffix;
        List<String> engineFiles = reader.getSegmentInfo()
            .files()
            .stream()
            .filter(fileName -> fileName.endsWith(underLineEngineSuffix))
            .sorted(Comparator.comparingInt(String::length))
            .collect(Collectors.toList());
        return engineFiles;
    }

    /**
     * Execute exact search for the given matched doc ids and return the results as a map of docId to score.
     *
     * @param leafReaderContext The leaf reader context for the current segment.
     * @param matchSet The filterIds to search for.
     * @param isParentHits Whether the matchedDocs contains parent ids or child ids.
     * @param k The number of results to return.
     * @return Map of docId to score for the exact search results.
     * @throws IOException If an error occurs during the search.
     */
    public Map<Integer, Float> exactSearch(final LeafReaderContext leafReaderContext, final BitSet matchSet, boolean isParentHits, int k)
        throws IOException {
        return exactSearcher.searchLeaf(leafReaderContext, matchSet, knnQuery, k, isParentHits);
    }

    @Override
    public boolean isCacheable(LeafReaderContext context) {
        return true;
    }

    public static float normalizeScore(float score) {
        if (score >= 0) return 1 / (1 + score);
        return -score + 1;
    }

    private boolean canDoExactSearch(final int filterIdsCount) {
        log.debug(
            "Info for doing exact search filterIdsLength : {}, Threshold value: {}",
            filterIdsCount,
            KNNSettings.getFilteredExactSearchThreshold(knnQuery.getIndexName())
        );
        if (knnQuery.getRadius() != null) {
            return false;
        }
        int filterThresholdValue = KNNSettings.getFilteredExactSearchThreshold(knnQuery.getIndexName());
        // Refer this GitHub around more details https://github.com/opensearch-project/k-NN/issues/1049 on the logic
        if (filterIdsCount <= knnQuery.getK()) {
            return true;
        }
        // See user has defined Exact Search filtered threshold. if yes, then use that setting.
        if (isExactSearchThresholdSettingSet(filterThresholdValue)) {
            return filterThresholdValue >= filterIdsCount;
        }

        // if no setting is set, then use the default max distance computation value to see if we can do exact search.
        /**
         * TODO we can have a different MAX_DISTANCE_COMPUTATIONS for binary index as computation cost for binary index
         * is cheaper than computation cost for non binary vector
         */
        return KNNConstants.MAX_DISTANCE_COMPUTATIONS >= filterIdsCount * (knnQuery.getVectorDataType() == VectorDataType.FLOAT
            ? knnQuery.getQueryVector().length
            : knnQuery.getByteQueryVector().length);
    }

    /**
     *  This function validates if {@link KNNSettings#ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD} is set or not. This
     *  is done by validating if the setting value is equal to the default value.
     * @param filterThresholdValue value of the Index Setting: {@link KNNSettings#ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD_SETTING}
     * @return boolean true if the setting is set.
     */
    private boolean isExactSearchThresholdSettingSet(int filterThresholdValue) {
        return filterThresholdValue != KNNSettings.ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD_DEFAULT_VALUE;
    }

    /**
     * This condition mainly checks during filtered search we have more than K elements in filterIds but the ANN
     * doesn't yeild K nearest neighbors.
     * @param filterIdsCount count of filtered Doc ids
     * @param annResultCount Count of Nearest Neighbours we got after doing filtered ANN Search.
     * @return boolean - true if exactSearch needs to be done after ANNSearch.
     */
    private boolean canDoExactSearchAfterANNSearch(final int filterIdsCount, final int annResultCount) {
        return filterWeight != null && filterIdsCount >= knnQuery.getK() && knnQuery.getK() > annResultCount;
    }

    // TODO: this will eventually return more types than just byte
    private byte[] getQuantizedVector(QuantizationParams quantizationParams, SegmentReader reader, FieldInfo fieldInfo) throws IOException {
        if (quantizationParams != null) {
            QuantizationConfigKNNCollector tempCollector = new QuantizationConfigKNNCollector();
            reader.searchNearestVectors(knnQuery.getField(), new float[0], tempCollector, null);
            if (tempCollector.getQuantizationState() == null) {
                throw new IllegalStateException(String.format("No quantization state found for field %s", fieldInfo.getName()));
            }
            QuantizationOutput quantizationOutput = quantizationService.createQuantizationOutput(quantizationParams);
            // TODO: In the future, byte array will not be the only output type from this method
            return (byte[]) quantizationService.quantize(
                tempCollector.getQuantizationState(),
                knnQuery.getQueryVector(),
                quantizationOutput
            );
        }
        return null;
    }
}
