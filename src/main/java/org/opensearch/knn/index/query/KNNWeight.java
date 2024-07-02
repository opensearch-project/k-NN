/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import lombok.extern.log4j.Log4j2;
import org.apache.commons.lang.StringUtils;
import org.apache.lucene.index.DocValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FilterLeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.Explanation;
import org.apache.lucene.search.FilteredDocIdSetIterator;
import org.apache.lucene.search.HitQueue;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.Weight;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.FilterDirectory;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.BitSetIterator;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.DocIdSetBuilder;
import org.apache.lucene.util.FixedBitSet;
import org.opensearch.common.io.PathUtils;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.memory.NativeMemoryAllocation;
import org.opensearch.knn.index.memory.NativeMemoryCacheManager;
import org.opensearch.knn.index.memory.NativeMemoryEntryContext;
import org.opensearch.knn.index.memory.NativeMemoryLoadStrategy;
import org.opensearch.knn.index.query.filtered.FilteredIdsKNNIterator;
import org.opensearch.knn.index.query.filtered.NestedFilteredIdsKNNIterator;
import org.opensearch.knn.index.util.FieldInfoExtractor;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.indices.ModelUtil;
import org.opensearch.knn.jni.JNIService;
import org.opensearch.knn.plugin.stats.KNNCounter;
import org.opensearch.knn.quantization.QuantizationManager;
import org.opensearch.knn.quantization.enums.SQTypes;
import org.opensearch.knn.quantization.models.quantizationParams.QuantizationParams;
import org.opensearch.knn.quantization.models.quantizationParams.SQParams;
import org.opensearch.knn.quantization.quantizer.Quantizer;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.stream.Collectors;

import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.MODEL_ID;
import static org.opensearch.knn.common.KNNConstants.SPACE_TYPE;
import static org.opensearch.knn.index.IndexUtil.getParametersAtLoading;
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

    public KNNWeight(KNNQuery query, float boost) {
        super(query);
        this.knnQuery = query;
        this.boost = boost;
        this.nativeMemoryCacheManager = NativeMemoryCacheManager.getInstance();
        this.filterWeight = null;
    }

    public KNNWeight(KNNQuery query, float boost, Weight filterWeight) {
        super(query);
        this.knnQuery = query;
        this.boost = boost;
        this.nativeMemoryCacheManager = NativeMemoryCacheManager.getInstance();
        this.filterWeight = filterWeight;
    }

    public static void initialize(ModelDao modelDao) {
        KNNWeight.modelDao = modelDao;
    }

    @Override
    public Explanation explain(LeafReaderContext context, int doc) {
        return Explanation.match(1.0f, "No Explanation");
    }

    @Override
    public Scorer scorer(LeafReaderContext context) throws IOException {
        // print all the vectors in the segment using Float Vector Values.
        // FloatVectorValues floatVectorValues = context.reader().getFloatVectorValues(knnQuery.getField());
        // if (floatVectorValues == null) {
        // log.info("floatVectorValues is null");
        // } else {
        // int maxDoc = context.reader().maxDoc();
        // for (int i = 0; i < maxDoc; i++) {
        // int docId = floatVectorValues.advance(i);
        // float[] vector = floatVectorValues.vectorValue();
        // log.info("docId: {}, vector: {}", docId, Arrays.toString(vector));
        // }
        // }

        final BitSet filterBitSet = getFilteredDocsBitSet(context);
        int cardinality = filterBitSet.cardinality();
        // We don't need to go to JNI layer if no documents are found which satisfy the filters
        // We should give this condition a deeper look that where it should be placed. For now I feel this is a good
        // place,
        if (filterWeight != null && cardinality == 0) {
            return KNNScorer.emptyScorer(this);
        }
        final Map<Integer, Float> docIdsToScoreMap = new HashMap<>();

        /*
         * The idea for this optimization is to get K results, we need to atleast look at K vectors in the HNSW graph
         * . Hence, if filtered results are less than K and filter query is present we should shift to exact search.
         * This improves the recall.
         */
        if (filterWeight != null && canDoExactSearch(cardinality)) {
            docIdsToScoreMap.putAll(doExactSearch(context, filterBitSet, cardinality));
        } else {
            Map<Integer, Float> annResults = doANNSearch(context, filterBitSet, cardinality);
            if (annResults == null) {
                return null;
            }
            if (canDoExactSearchAfterANNSearch(cardinality, annResults.size())) {
                log.debug(
                    "Doing ExactSearch after doing ANNSearch as the number of documents returned are less than "
                        + "K, even when we have more than K filtered Ids. K: {}, ANNResults: {}, filteredIdCount: {}",
                    knnQuery.getK(),
                    annResults.size(),
                    cardinality
                );
                annResults = doExactSearch(context, filterBitSet, cardinality);
            }
            docIdsToScoreMap.putAll(annResults);
        }
        if (docIdsToScoreMap.isEmpty()) {
            return KNNScorer.emptyScorer(this);
        }
        return convertSearchResponseToScorer(docIdsToScoreMap);
    }

    private QuantizationParams getQuantizationParams() {
        // Implement this method to return appropriate quantization parameters based on your use case
        return new SQParams(SQTypes.ONE_BIT); // Example, modify as needed
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

    private Map<Integer, Float> doANNSearch(final LeafReaderContext context, final BitSet filterIdsBitSet, final int cardinality)
        throws IOException {
        SegmentReader reader = (SegmentReader) FilterLeafReader.unwrap(context.reader());
        String directory = ((FSDirectory) FilterDirectory.unwrap(reader.directory())).getDirectory().toString();

        QuantizationParams params = getQuantizationParams(); // Implement this method to get appropriate params
        Quantizer<float[], byte[]> quantizer = (Quantizer<float[], byte[]>) QuantizationManager.getInstance().getQuantizer(params);

        FieldInfo fieldInfo = reader.getFieldInfos().fieldInfo(knnQuery.getField());

        if (fieldInfo == null) {
            log.debug("[KNN] Field info not found for {}:{}", knnQuery.getField(), reader.getSegmentName());
            return null;
        }

        KNNEngine knnEngine;
        SpaceType spaceType;

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
        } else {
            String engineName = fieldInfo.attributes().getOrDefault(KNN_ENGINE, KNNEngine.NMSLIB.getName());
            knnEngine = KNNEngine.getEngine(engineName);
            String spaceTypeName = fieldInfo.attributes().getOrDefault(SPACE_TYPE, SpaceType.L2.getValue());
            spaceType = SpaceType.getSpace(spaceTypeName);
        }

        /*
         * In case of compound file, extension would be <engine-extension> + c otherwise <engine-extension>
         */
        String engineExtension = reader.getSegmentInfo().info.getUseCompoundFile()
            ? knnEngine.getExtension() + KNNConstants.COMPOUND_EXTENSION
            : knnEngine.getExtension();
        String engineSuffix = knnQuery.getField() + engineExtension;
        List<String> engineFiles = reader.getSegmentInfo()
            .files()
            .stream()
            .filter(fileName -> fileName.endsWith(engineSuffix))
            .collect(Collectors.toList());

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
                        "B" + FieldInfoExtractor.getIndexDescription(fieldInfo)
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
            byte[] quantizedVector = quantizer.quantize(knnQuery.getQueryVector()).getQuantizedVector();
            if (knnQuery.getK() > 0) {
                results = JNIService.queryBinaryIndex(
                    indexAllocation.getMemoryAddress(),
                        quantizedVector,
                    knnQuery.getK(),
                    knnEngine,
                    filterIds,
                    filterType.getValue(),
                    parentIds
                );
            } else {
                results = JNIService.radiusQueryIndex(
                    indexAllocation.getMemoryAddress(),
                    knnQuery.getQueryVector(),
                    knnQuery.getRadius(),
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

    private Map<Integer, Float> doExactSearch(final LeafReaderContext leafReaderContext, final BitSet filterIdsBitSet, int cardinality) {
        try {
            // Creating min heap and init with MAX DocID and Score as -INF.
            final HitQueue queue = new HitQueue(Math.min(this.knnQuery.getK(), cardinality), true);
            ScoreDoc topDoc = queue.top();
            final Map<Integer, Float> docToScore = new HashMap<>();
            FilteredIdsKNNIterator iterator = getFilteredKNNIterator(leafReaderContext, filterIdsBitSet);
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
        } catch (Exception e) {
            log.error("Error while getting the doc values to do the k-NN Search for query : {}", this.knnQuery, e);
        }
        return Collections.emptyMap();
    }

    private FilteredIdsKNNIterator getFilteredKNNIterator(final LeafReaderContext leafReaderContext, final BitSet filterIdsBitSet)
        throws IOException {
        final SegmentReader reader = (SegmentReader) FilterLeafReader.unwrap(leafReaderContext.reader());
        final FieldInfo fieldInfo = reader.getFieldInfos().fieldInfo(knnQuery.getField());
        DocIdSetIterator docIdSetIterator;
        if (fieldInfo.getVectorDimension() > 0) {
            docIdSetIterator = reader.getFloatVectorValues(knnQuery.getField());
        } else {
            docIdSetIterator = DocValues.getBinary(leafReaderContext.reader(), fieldInfo.getName());
        }

        final SpaceType spaceType = getSpaceType(fieldInfo);
        return knnQuery.getParentsFilter() == null
            ? new FilteredIdsKNNIterator(filterIdsBitSet, knnQuery.getQueryVector(), docIdSetIterator, spaceType)
            : new NestedFilteredIdsKNNIterator(
                filterIdsBitSet,
                knnQuery.getQueryVector(),
                docIdSetIterator,
                spaceType,
                knnQuery.getParentsFilter().getBitSet(leafReaderContext)
            );
    }

    private Scorer convertSearchResponseToScorer(final Map<Integer, Float> docsToScore) throws IOException {
        final int maxDoc = Collections.max(docsToScore.keySet()) + 1;
        final DocIdSetBuilder docIdSetBuilder = new DocIdSetBuilder(maxDoc);
        // The docIdSetIterator will contain the docids of the returned results. So, before adding results to
        // the builder, we can grow to docsToScore.size()
        final DocIdSetBuilder.BulkAdder setAdder = docIdSetBuilder.grow(docsToScore.size());
        docsToScore.keySet().forEach(setAdder::add);
        final DocIdSetIterator docIdSetIter = docIdSetBuilder.build().iterator();
        return new KNNScorer(this, docIdSetIter, docsToScore, boost);
    }

    @Override
    public boolean isCacheable(LeafReaderContext context) {
        return true;
    }

    public static float normalizeScore(float score) {
        if (score >= 0) return 1 / (1 + score);
        return -score + 1;
    }

    private SpaceType getSpaceType(final FieldInfo fieldInfo) {
        final String spaceTypeString = fieldInfo.getAttribute(SPACE_TYPE);
        if (StringUtils.isNotEmpty(spaceTypeString)) {
            return SpaceType.getSpace(spaceTypeString);
        }

        final String modelId = fieldInfo.getAttribute(MODEL_ID);
        if (StringUtils.isNotEmpty(modelId)) {
            ModelMetadata modelMetadata = modelDao.getMetadata(modelId);
            return modelMetadata.getSpaceType();
        }
        throw new IllegalArgumentException(
            String.format(Locale.ROOT, "Unable to find the Space Type from Field Info attribute for field %s", fieldInfo.getName())
        );
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
        return KNNConstants.MAX_DISTANCE_COMPUTATIONS >= filterIdsCount * knnQuery.getQueryVector().length;
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
}
