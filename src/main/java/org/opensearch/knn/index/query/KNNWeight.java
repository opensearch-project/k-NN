/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import com.google.common.annotations.VisibleForTesting;
import lombok.Getter;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.Explanation;
import org.apache.lucene.search.FilteredDocIdSetIterator;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.ScorerSupplier;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.search.Weight;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.BitSetIterator;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;
import org.opensearch.common.Nullable;
import org.opensearch.common.StopWatch;
import org.opensearch.common.lucene.Lucene;
import org.opensearch.knn.common.FieldInfoExtractor;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.util.KNNCodecUtil;
import org.opensearch.knn.index.codec.util.NativeMemoryCacheKeyHelper;
import org.opensearch.knn.index.memory.NativeMemoryAllocation;
import org.opensearch.knn.index.memory.NativeMemoryCacheManager;
import org.opensearch.knn.index.memory.NativeMemoryEntryContext;
import org.opensearch.knn.index.memory.NativeMemoryLoadStrategy;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.quantizationservice.QuantizationService;
import org.opensearch.knn.index.query.ExactSearcher.ExactSearcherContext.ExactSearcherContextBuilder;
import org.opensearch.knn.index.query.explain.KnnExplanation;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.indices.ModelUtil;
import org.opensearch.knn.jni.JNIService;
import org.opensearch.knn.plugin.stats.KNNCounter;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
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
 * {@link KNNWeight} serves as a template for implementing approximate nearest neighbor (ANN)
 * and radius search over a native index type, such as Faiss.
 * <p>
 * It encapsulates shared logic such as quantization, extracting relevant search information,
 * and determining whether to fall back to exact search based on configuration settings.
 * <p>
 * This abstract class is extended by two concrete implementations:
 * <ol>
 *   <li>{@link DefaultKNNWeight} – delegates vector search to a native library via JNI
 *       (e.g., Faiss written in C++).</li>
 *   <li>{@link org.opensearch.knn.index.query.memoryoptsearch.MemoryOptimizedKNNWeight} – uses the Lucene API to perform vector search
 *       in a memory-efficient manner.</li>
 * </ol>
 * Although their internal mechanisms for retrieving KNN results differ, both are designed
 * to produce nearly identical recall while subjected to a common KNN search workflow. e.g. with the identical settings, regardless of which
 * one is being used, user will get the same results.
 */
@Log4j2
public abstract class KNNWeight extends Weight {
    protected static final TopDocs EMPTY_TOPDOCS = new TopDocs(new TotalHits(0, TotalHits.Relation.EQUAL_TO), new ScoreDoc[0]);
    private static ModelDao modelDao;
    private static ExactSearcher DEFAULT_EXACT_SEARCHER;

    protected final KNNQuery knnQuery;
    private final float boost;

    @Getter
    private final Weight filterWeight;
    private final ExactSearcher exactSearcher;

    protected final QuantizationService quantizationService;
    private final KnnExplanation knnExplanation;

    public KNNWeight(KNNQuery query, float boost) {
        this(query, boost, null);
    }

    public KNNWeight(KNNQuery query, float boost, Weight filterWeight) {
        super(query);
        this.knnQuery = query;
        this.boost = boost;
        this.filterWeight = filterWeight;
        this.exactSearcher = DEFAULT_EXACT_SEARCHER;
        this.quantizationService = QuantizationService.getInstance();
        this.knnExplanation = new KnnExplanation();
    }

    public static void initialize(ModelDao modelDao) {
        initialize(modelDao, new ExactSearcher(modelDao));
    }

    @VisibleForTesting
    static void initialize(ModelDao modelDao, ExactSearcher exactSearcher) {
        KNNWeight.modelDao = modelDao;
        KNNWeight.DEFAULT_EXACT_SEARCHER = exactSearcher;
    }

    @VisibleForTesting
    KnnExplanation getKnnExplanation() {
        return knnExplanation;
    }

    @Override
    // This method is called in case of Radial-Search
    public Explanation explain(LeafReaderContext context, int doc) {
        return explain(context, doc, 0);
    }

    // This method is called for ANN/Exact/Disk-based/Efficient-filtering search
    public Explanation explain(LeafReaderContext context, int doc, float score) {
        knnQuery.setExplain(true);
        try {
            final Scorer knnScorer = getOrCreateKnnScorer(context);
            // calculate score only when its 0 as for disk-based search,
            // score will be passed from the caller and there is no need to re-compute the score
            if (score == 0) {
                score = getKnnScore(knnScorer, doc);
            }
        } catch (IOException e) {
            throw new RuntimeException(String.format("Error while explaining KNN score for doc [%d], score [%f]", doc, score), e);
        }
        final String highLevelExplanation = getHighLevelExplanation();
        final StringBuilder leafLevelExplanation = getLeafLevelExplanation(context);

        final SegmentReader reader = Lucene.segmentReader(context.reader());
        final FieldInfo fieldInfo = FieldInfoExtractor.getFieldInfo(reader, knnQuery.getField());
        if (fieldInfo == null) {
            return Explanation.match(score, highLevelExplanation, Explanation.match(score, leafLevelExplanation.toString()));
        }
        final SpaceType spaceType = FieldInfoExtractor.getSpaceType(modelDao, fieldInfo);
        leafLevelExplanation.append(", spaceType = ").append(spaceType.getValue());

        final Float rawScore = knnExplanation.getRawScore(doc);
        Explanation rawScoreDetail = null;
        if (rawScore != null && knnQuery.getRescoreContext() == null) {
            leafLevelExplanation.append(" where score is computed as ")
                .append(spaceType.explainScoreTranslation(rawScore))
                .append(" from:");
            final String detailSuffix;
            if (knnQuery.isMemoryOptimizedSearch()) {
                detailSuffix = " memory optimized searching logic";
            } else {
                detailSuffix = " library";
            }
            rawScoreDetail = Explanation.match(
                rawScore,
                "rawScore, returned from " + FieldInfoExtractor.extractKNNEngine(fieldInfo) + detailSuffix
            );
        }

        return rawScoreDetail != null
            ? Explanation.match(score, highLevelExplanation, Explanation.match(score, leafLevelExplanation.toString(), rawScoreDetail))
            : Explanation.match(score, highLevelExplanation, Explanation.match(score, leafLevelExplanation.toString()));
    }

    private StringBuilder getLeafLevelExplanation(LeafReaderContext context) {
        int filterThresholdValue = KNNSettings.getFilteredExactSearchThreshold(knnQuery.getIndexName());
        int cardinality = knnExplanation.getCardinality();
        final StringBuilder sb = new StringBuilder("the type of knn search executed at leaf was ");
        if (filterWeight != null) {
            if (isFilterIdCountLessThanK(cardinality)) {
                sb.append(KNNConstants.EXACT_SEARCH)
                    .append(" since filteredIds = ")
                    .append(cardinality)
                    .append(" is less than or equal to K = ")
                    .append(knnQuery.getK());
            } else if (isExactSearchThresholdSettingSet(filterThresholdValue) && (filterThresholdValue >= cardinality)) {
                sb.append(KNNConstants.EXACT_SEARCH)
                    .append(" since filtered threshold value = ")
                    .append(filterThresholdValue)
                    .append(" is greater than or equal to cardinality = ")
                    .append(cardinality);
            } else if (!isExactSearchThresholdSettingSet(filterThresholdValue) && isMDCGreaterThanFilterIdCnt(cardinality)) {
                sb.append(KNNConstants.EXACT_SEARCH)
                    .append(" since max distance computation = ")
                    .append(KNNConstants.MAX_DISTANCE_COMPUTATIONS)
                    .append(" is greater than or equal to cardinality = ")
                    .append(cardinality);
            }
        }
        final Integer annResult = knnExplanation.getAnnResult(context.id());
        if (annResult != null && annResult == 0 && isMissingNativeEngineFiles(context)) {
            sb.append(KNNConstants.EXACT_SEARCH).append(" since no native engine files are available");
        }
        if (annResult != null && isFilteredExactSearchRequireAfterANNSearch(cardinality, annResult)) {
            sb.append(KNNConstants.EXACT_SEARCH)
                .append(" since the number of documents returned are less than K = ")
                .append(knnQuery.getK())
                .append(" and there are more than K filtered Ids = ")
                .append(cardinality);
        }
        if (annResult != null && annResult > 0 && !isFilteredExactSearchRequireAfterANNSearch(cardinality, annResult)) {
            sb.append(KNNConstants.ANN_SEARCH);
        }
        sb.append(" with vectorDataType = ").append(knnQuery.getVectorDataType());
        return sb;
    }

    private String getHighLevelExplanation() {
        final StringBuilder sb = new StringBuilder("the type of knn search executed was ");
        if (knnQuery.getRescoreContext() != null) {
            sb.append(buildDiskBasedSearchExplanation());
        } else if (knnQuery.getRadius() != null) {
            sb.append(KNNConstants.RADIAL_SEARCH).append(" with the radius of ").append(knnQuery.getRadius());
        } else {
            sb.append(KNNConstants.ANN_SEARCH);
        }
        return sb.toString();
    }

    private String buildDiskBasedSearchExplanation() {
        StringBuilder sb = new StringBuilder(KNNConstants.DISK_BASED_SEARCH);
        boolean isShardLevelRescoringDisabled = KNNSettings.isShardLevelRescoringDisabledForDiskBasedVector(knnQuery.getIndexName());
        if (!knnQuery.getRescoreContext().isRescoreEnabled()) {
            isShardLevelRescoringDisabled = true;
        }
        int dimension = knnQuery.getQueryVector().length;
        int firstPassK = knnQuery.getRescoreContext().getFirstPassK(knnQuery.getK(), isShardLevelRescoringDisabled, dimension);
        sb.append(" and the first pass k was ")
            .append(firstPassK)
            .append(" with vector dimension of ")
            .append(dimension)
            .append(", over sampling factor of ")
            .append(knnQuery.getRescoreContext().getOversampleFactor());
        if (isShardLevelRescoringDisabled) {
            sb.append(", shard level rescoring disabled");
        } else {
            sb.append(", shard level rescoring enabled");
        }
        return sb.toString();
    }

    private Scorer getOrCreateKnnScorer(LeafReaderContext context) throws IOException {
        // First try to get the cached scorer
        Scorer scorer = knnExplanation.getKnnScorer(context);

        // If no cached scorer exists, create and cache a new one
        if (scorer == null) {
            scorer = (KNNScorer) scorer(context);
            knnExplanation.addKnnScorer(context, scorer);
        }

        return scorer;
    }

    private float getKnnScore(Scorer knnScorer, int doc) throws IOException {
        return (knnScorer.iterator().advance(doc) == doc) ? knnScorer.score() : 0;
    }

    @Override
    public ScorerSupplier scorerSupplier(LeafReaderContext context) {
        return new ScorerSupplier() {
            long cost = -1L;

            @Override
            public Scorer get(long leadCost) throws IOException {
                final TopDocs topDocs = searchLeaf(context, knnQuery.getK()).getResult();
                cost = topDocs.scoreDocs.length;
                if (cost == 0) {
                    return KNNScorer.emptyScorer();
                }
                return new KNNScorer(topDocs, boost);
            }

            @Override
            public long cost() {
                // Estimate the cost of the scoring operation, if applicable.
                return cost == -1L ? knnQuery.getK() : cost;
            }
        };
    }

    /**
     * Executes k nearest neighbor search for a segment to get the top K results
     * This is made public purely to be able to be reused in {@link org.opensearch.knn.index.query.nativelib.NativeEngineKnnVectorQuery}
     *
     * @param context LeafReaderContext
     * @param k Number of results to return
     * @return A Map of docId to scores for top k results
     */
    public PerLeafResult searchLeaf(LeafReaderContext context, int k) throws IOException {
        final SegmentReader reader = Lucene.segmentReader(context.reader());
        final String segmentName = reader.getSegmentName();

        StopWatch stopWatch = startStopWatch();
        final BitSet filterBitSet = getFilteredDocsBitSet(context);
        stopStopWatchAndLog(stopWatch, "FilterBitSet creation", segmentName);

        final int maxDoc = context.reader().maxDoc();
        int cardinality = filterBitSet.cardinality();
        // We don't need to go to JNI layer if no documents are found which satisfy the filters
        // We should give this condition a deeper look that where it should be placed. For now I feel this is a good
        // place,
        if (filterWeight != null && cardinality == 0) {
            return PerLeafResult.EMPTY_RESULT;
        }
        if (knnQuery.isExplain()) {
            knnExplanation.setCardinality(cardinality);
        }

        /*
         * The idea for this optimization is to get K results, we need to at least look at K vectors in the HNSW graph
         * . Hence, if filtered results are less than K and filter query is present we should shift to exact search.
         * This improves the recall.
         */
        if (isFilteredExactSearchPreferred(cardinality)) {
            TopDocs result = doExactSearch(context, new BitSetIterator(filterBitSet, cardinality), cardinality, k);
            return new PerLeafResult(filterWeight == null ? null : filterBitSet, result);
        }

        /*
         * If filters match all docs in this segment, then null should be passed as filterBitSet
         * so that it will not do a bitset look up in bottom search layer.
         */
        final BitSet annFilter = (filterWeight != null && cardinality == maxDoc) ? null : filterBitSet;

        StopWatch annStopWatch = startStopWatch();
        final TopDocs topDocs = approximateSearch(context, annFilter, cardinality, k);
        stopStopWatchAndLog(annStopWatch, "ANN search", segmentName);
        if (knnQuery.isExplain()) {
            knnExplanation.addLeafResult(context.id(), topDocs.scoreDocs.length);
        }
        // See whether we have to perform exact search based on approx search results
        // This is required if there are no native engine files or if approximate search returned
        // results less than K, though we have more than k filtered docs
        if (isExactSearchRequire(context, cardinality, topDocs.scoreDocs.length)) {
            final BitSetIterator docs = filterWeight != null ? new BitSetIterator(filterBitSet, cardinality) : null;
            TopDocs result = doExactSearch(context, docs, cardinality, k);
            return new PerLeafResult(filterWeight == null ? null : filterBitSet, result);
        }
        return new PerLeafResult(filterWeight == null ? null : filterBitSet, topDocs);
    }

    private void stopStopWatchAndLog(@Nullable final StopWatch stopWatch, final String prefixMessage, String segmentName) {
        if (log.isDebugEnabled() && stopWatch != null) {
            stopWatch.stop();
            final String logMessage = prefixMessage + " shard: [{}], segment: [{}], field: [{}], time in nanos:[{}] ";
            log.debug(logMessage, knnQuery.getShardId(), segmentName, knnQuery.getField(), stopWatch.totalTime().nanos());
        }
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

    private TopDocs doExactSearch(
        final LeafReaderContext context,
        final DocIdSetIterator acceptedDocs,
        final long numberOfAcceptedDocs,
        final int k
    ) throws IOException {
        final ExactSearcherContextBuilder exactSearcherContextBuilder = ExactSearcher.ExactSearcherContext.builder()
            .parentsFilter(knnQuery.getParentsFilter())
            .k(k)
            // setting to true, so that if quantization details are present we want to do search on the quantized
            // vectors as this flow is used in first pass of search.
            .useQuantizedVectorsForSearch(true)
            .field(knnQuery.getField())
            .radius(knnQuery.getRadius())
            .matchedDocsIterator(acceptedDocs)
            .numberOfMatchedDocs(numberOfAcceptedDocs)
            .floatQueryVector(knnQuery.getQueryVector())
            .byteQueryVector(knnQuery.getByteQueryVector())
            .isMemoryOptimizedSearchEnabled(knnQuery.isMemoryOptimizedSearch());

        if (knnQuery.getContext() != null) {
            exactSearcherContextBuilder.maxResultWindow(knnQuery.getContext().getMaxResultWindow());
        }

        return exactSearch(context, exactSearcherContextBuilder.build());
    }

    private TopDocs approximateSearch(final LeafReaderContext context, final BitSet filterIdsBitSet, final int cardinality, final int k)
        throws IOException {
        final SegmentReader reader = Lucene.segmentReader(context.reader());
        FieldInfo fieldInfo = FieldInfoExtractor.getFieldInfo(reader, knnQuery.getField());

        if (fieldInfo == null) {
            log.debug("[KNN] Field info not found for {}:{}", knnQuery.getField(), reader.getSegmentName());
            return EMPTY_TOPDOCS;
        }

        KNNEngine knnEngine;
        SpaceType spaceType;
        VectorDataType vectorDataType;

        // Check if a modelId exists. If so, the space type and engine will need to be picked up from the model's
        // metadata.
        final String modelId = fieldInfo.getAttribute(MODEL_ID);
        if (modelId != null) {
            ModelMetadata modelMetadata = modelDao.getMetadata(modelId);
            if (!ModelUtil.isModelCreated(modelMetadata)) {
                throw new RuntimeException("Model \"" + modelId + "\" is not created.");
            }

            knnEngine = modelMetadata.getKnnEngine();
            spaceType = modelMetadata.getSpaceType();
            vectorDataType = modelMetadata.getVectorDataType();
        } else {
            final String engineName = fieldInfo.attributes().getOrDefault(KNN_ENGINE, KNNEngine.DEFAULT.getName());
            knnEngine = KNNEngine.getEngine(engineName);
            final String spaceTypeName = fieldInfo.attributes().getOrDefault(SPACE_TYPE, SpaceType.L2.getValue());
            spaceType = SpaceType.getSpace(spaceTypeName);
            vectorDataType = VectorDataType.get(
                fieldInfo.attributes().getOrDefault(VECTOR_DATA_TYPE_FIELD, VectorDataType.FLOAT.getValue())
            );
        }

        final SegmentLevelQuantizationInfo segmentLevelQuantizationInfo = SegmentLevelQuantizationInfo.build(
            reader,
            fieldInfo,
            knnQuery.getField(),
            reader.getSegmentInfo().info.getVersion()
        );

        List<String> engineFiles = KNNCodecUtil.getEngineFiles(knnEngine.getExtension(), knnQuery.getField(), reader.getSegmentInfo().info);
        if (engineFiles.isEmpty()) {
            log.debug("[KNN] No native engine files found for field {} for segment {}", knnQuery.getField(), reader.getSegmentName());
            return EMPTY_TOPDOCS;
        }

        // TODO: Change type of vector once more quantization methods are supported
        byte[] quantizedVector = null;
        if (SegmentLevelQuantizationUtil.isAdcEnabled(segmentLevelQuantizationInfo)) {
            SegmentLevelQuantizationUtil.transformVector(knnQuery.getQueryVector(), segmentLevelQuantizationInfo);
        } else {
            quantizedVector = SegmentLevelQuantizationUtil.quantizeVector(knnQuery.getQueryVector(), segmentLevelQuantizationInfo);
        }

        KNNCounter.GRAPH_QUERY_REQUESTS.increment();
        final TopDocs results = doANNSearch(
            context,
            reader,
            fieldInfo,
            spaceType,
            knnEngine,
            vectorDataType,
            quantizedVector,
            modelId,
            filterIdsBitSet,
            cardinality,
            k
        );

        if (results.scoreDocs.length == 0) {
            log.debug("[KNN] Query yielded 0 results");
            return EMPTY_TOPDOCS;
        }

        return results;
    }

    /**
     * As the name suggests, it performs an approximate nearest neighbor (ANN) search and returns the collected results as a {@code Map}.
     *
     * @param context LeafReaderContext.
     * @param reader SegmentReader having segment info and {@link org.apache.lucene.codecs.KnnVectorsReader}.
     * @param fieldInfo Target field info for searching.
     * @param spaceType Space type configured for the target field.
     * @param knnEngine Engine type configured for the target field.
     * @param vectorDataType Vector data type configured for the target field.
     * @param quantizedVector Quantized query vector if quantization is enabled for the target field. It can be null. Quantized query vector if quantization is enabled for the target field. It can be null. Quantized query vector if quantization is enabled for the target field. It can be null. Quantized query vector if quantization is enabled for the target field. It can be null. Quantized query vector if quantization is enabled for the target field. It can be null. Quantized query vector if quantization is enabled for the target field. It can be null.
     * @param modelId Model id. It can be null if the index for searching was not derived from a trained index.
     * @param filterIdsBitSet Bit set for filtering a valid document for collecting.
     * @param cardinality Cardinality of filtering bit set. It will be the total number of documents if no filtering presents.
     * @param k The number of documents to be collected
     * @return A table maps document id to its Lucene score.
     * @throws IOException
     */
    abstract protected TopDocs doANNSearch(
        final LeafReaderContext context,
        final SegmentReader reader,
        final FieldInfo fieldInfo,
        final SpaceType spaceType,
        final KNNEngine knnEngine,
        final VectorDataType vectorDataType,
        final byte[] quantizedVector,
        final String modelId,
        final BitSet filterIdsBitSet,
        final int cardinality,
        final int k
    ) throws IOException;

    protected void addExplainIfRequired(final KNNQueryResult[] results, final KNNEngine knnEngine, final SpaceType spaceType) {
        if (knnQuery.isExplain()) {
            Arrays.stream(results).forEach(result -> {
                if (KNNEngine.FAISS.getName().equals(knnEngine.getName()) && SpaceType.INNER_PRODUCT.equals(spaceType)) {
                    knnExplanation.addRawScore(result.getId(), -1 * result.getScore());
                } else {
                    knnExplanation.addRawScore(result.getId(), result.getScore());
                }
            });
        }
    }

    protected void addExplainIfRequired(final TopDocs results, final KNNEngine knnEngine, final SpaceType spaceType) {
        if (knnQuery.isExplain()) {
            Arrays.stream(results.scoreDocs).forEach(result -> {
                if (KNNEngine.FAISS.getName().equals(knnEngine.getName()) && SpaceType.INNER_PRODUCT.equals(spaceType)) {
                    knnExplanation.addRawScore(result.doc, -1 * result.score);
                } else {
                    knnExplanation.addRawScore(result.doc, result.score);
                }
            });
        }
    }

    /**
     * Execute exact search for the given matched doc ids and return the results as a map of docId to score.
     * @return Map of docId to score for the exact search results.
     * @throws IOException If an error occurs during the search.
     */
    public TopDocs exactSearch(final LeafReaderContext leafReaderContext, final ExactSearcher.ExactSearcherContext exactSearcherContext)
        throws IOException {
        StopWatch stopWatch = startStopWatch();
        TopDocs exactSearchResults = exactSearcher.searchLeaf(leafReaderContext, exactSearcherContext);
        final SegmentReader reader = Lucene.segmentReader(leafReaderContext.reader());
        stopStopWatchAndLog(stopWatch, "Exact search", reader.getSegmentName());
        return exactSearchResults;
    }

    @Override
    public boolean isCacheable(LeafReaderContext context) {
        return true;
    }

    public static float normalizeScore(float score) {
        if (score >= 0) return 1 / (1 + score);
        return -score + 1;
    }

    private boolean isFilteredExactSearchPreferred(final int filterIdsCount) {
        if (filterWeight == null) {
            return false;
        }
        log.debug(
            "Info for doing exact search filterIdsLength : {}, Threshold value: {}",
            filterIdsCount,
            KNNSettings.getFilteredExactSearchThreshold(knnQuery.getIndexName())
        );
        int filterThresholdValue = KNNSettings.getFilteredExactSearchThreshold(knnQuery.getIndexName());
        // Refer this GitHub around more details https://github.com/opensearch-project/k-NN/issues/1049 on the logic
        if (isFilterIdCountLessThanK(filterIdsCount)) return true;
        // See user has defined Exact Search filtered threshold. if yes, then use that setting.
        if (isExactSearchThresholdSettingSet(filterThresholdValue)) {
            if (filterThresholdValue >= filterIdsCount) {
                return true;
            }
            return false;
        }

        // if no setting is set, then use the default max distance computation value to see if we can do exact search.
        /**
         * TODO we can have a different MAX_DISTANCE_COMPUTATIONS for binary index as computation cost for binary index
         * is cheaper than computation cost for non binary vector
         */
        return isMDCGreaterThanFilterIdCnt(filterIdsCount);
    }

    private boolean isMDCGreaterThanFilterIdCnt(int filterIdsCount) {
        return KNNConstants.MAX_DISTANCE_COMPUTATIONS >= filterIdsCount * (knnQuery.getVectorDataType() == VectorDataType.FLOAT
            ? knnQuery.getQueryVector().length
            : knnQuery.getByteQueryVector().length);
    }

    private boolean isFilterIdCountLessThanK(int filterIdsCount) {
        return knnQuery.getRadius() == null && filterIdsCount <= knnQuery.getK();
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
     * This condition mainly checks whether exact search should be performed or not
     * @param context LeafReaderContext
     * @param filterIdsCount count of filtered Doc ids
     * @param annResultCount Count of Nearest Neighbours we got after doing filtered ANN Search.
     * @return boolean - true if exactSearch needs to be done after ANNSearch.
     */
    private boolean isExactSearchRequire(final LeafReaderContext context, final int filterIdsCount, final int annResultCount) {
        if (annResultCount == 0 && isMissingNativeEngineFiles(context)) {
            log.debug("Perform exact search after approximate search since no native engine files are available");
            return true;
        }
        if (isFilteredExactSearchRequireAfterANNSearch(filterIdsCount, annResultCount)) {
            log.debug(
                "Doing ExactSearch after doing ANNSearch as the number of documents returned are less than "
                    + "K, even when we have more than K filtered Ids. K: {}, ANNResults: {}, filteredIdCount: {}",
                this.knnQuery.getK(),
                annResultCount,
                filterIdsCount
            );
            return true;
        }
        return false;
    }

    /**
     * This condition mainly checks during filtered search we have more than K elements in filterIds but the ANN
     * doesn't yield K nearest neighbors.
     * @param filterIdsCount count of filtered Doc ids
     * @param annResultCount Count of Nearest Neighbours we got after doing filtered ANN Search.
     * @return boolean - true if exactSearch needs to be done after ANNSearch.
     */
    private boolean isFilteredExactSearchRequireAfterANNSearch(final int filterIdsCount, final int annResultCount) {
        return filterWeight != null && filterIdsCount >= knnQuery.getK() && knnQuery.getK() > annResultCount;
    }

    /**
     * This condition mainly checks whether segments has native engine files or not
     * @return boolean - false if exactSearch needs to be done since no native engine files are in segments.
     */
    private boolean isMissingNativeEngineFiles(LeafReaderContext context) {
        final SegmentReader reader = Lucene.segmentReader(context.reader());
        final FieldInfo fieldInfo = FieldInfoExtractor.getFieldInfo(reader, knnQuery.getField());
        // if segment has no documents with at least 1 vector field, field info will be null
        if (fieldInfo == null) {
            return false;
        }
        final KNNEngine knnEngine = FieldInfoExtractor.extractKNNEngine(fieldInfo);
        final List<String> engineFiles = KNNCodecUtil.getEngineFiles(
            knnEngine.getExtension(),
            knnQuery.getField(),
            reader.getSegmentInfo().info
        );
        return engineFiles.isEmpty();
    }

    private StopWatch startStopWatch() {
        if (log.isDebugEnabled()) {
            return new StopWatch().start();
        }
        return null;
    }

    protected int[] getParentIdsArray(final LeafReaderContext context) throws IOException {
        if (knnQuery.getParentsFilter() == null) {
            return null;
        }
        return bitSetToIntArray(knnQuery.getParentsFilter().getBitSet(context));
    }

    private static int[] bitSetToIntArray(final BitSet bitSet) {
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
}
