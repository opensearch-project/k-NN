/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.DocValues;
import org.apache.lucene.search.FilteredDocIdSetIterator;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.BitSetIterator;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.FixedBitSet;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.codec.util.KNNVectorSerializer;
import org.opensearch.knn.index.codec.util.KNNVectorSerializerFactory;
import org.opensearch.knn.jni.JNIService;
import org.opensearch.knn.index.memory.NativeMemoryAllocation;
import org.opensearch.knn.index.memory.NativeMemoryCacheManager;
import org.opensearch.knn.index.memory.NativeMemoryEntryContext;
import org.opensearch.knn.index.memory.NativeMemoryLoadStrategy;
import org.opensearch.knn.index.util.KNNEngine;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FilterLeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.Explanation;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.Weight;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.FilterDirectory;
import org.apache.lucene.util.DocIdSetBuilder;
import org.opensearch.common.io.PathUtils;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.plugin.stats.KNNCounter;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
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
        final int[] filterIdsArray = getFilterIdsArray(context);
        // We don't need to go to JNI layer if no documents are found which satisfy the filters
        // We should give this condition a deeper look that where it should be placed. For now I feel this is a good
        // place,
        if (filterWeight != null && filterIdsArray.length == 0) {
            return KNNScorer.emptyScorer(this);
        }
        final Map<Integer, Float> docIdsToScoreMap = new HashMap<>();

        /*
         * The idea for this optimization is to get K results, we need to atleast look at K vectors in the HNSW graph
         * . Hence, if filtered results are less than K and filter query is present we should shift to exact search.
         * This improves the recall.
         */
        if (filterWeight != null && filterIdsArray.length <= knnQuery.getK()) {
            docIdsToScoreMap.putAll(doExactSearch(context, filterIdsArray));
        } else {
            final Map<Integer, Float> annResults = doANNSearch(context, filterIdsArray);
            if (annResults == null) {
                return null;
            }
            docIdsToScoreMap.putAll(annResults);
        }
        return convertSearchResponseToScorer(docIdsToScoreMap);
    }

    private BitSet getFilteredDocsBitSet(final LeafReaderContext ctx, final Weight filterWeight) throws IOException {
        final Bits liveDocs = ctx.reader().getLiveDocs();
        final int maxDoc = ctx.reader().maxDoc();

        final Scorer scorer = filterWeight.scorer(ctx);
        if (scorer == null) {
            return new FixedBitSet(0);
        }

        final BitSet acceptDocs = createBitSet(scorer.iterator(), liveDocs, maxDoc);
        // TODO: Based on this cost shift to exact search, because even in ANN search you have to calculate the
        // distance for K vectors. This can avoid calls to native layer and save some latency.
        final int cost = acceptDocs.cardinality();
        log.debug("Number of docs valid for filter is = Cost for filtered k-nn is : {}", cost);
        return acceptDocs;
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

    private int[] getFilterIdsArray(final LeafReaderContext context) throws IOException {
        if (filterWeight == null) {
            return new int[0];
        }
        final BitSet filteredDocsBitSet = getFilteredDocsBitSet(context, this.filterWeight);
        final int[] filteredIds = new int[filteredDocsBitSet.cardinality()];
        int filteredIdsIndex = 0;
        int docId = 0;
        while (true) {
            docId = filteredDocsBitSet.nextSetBit(docId);
            if (docId == DocIdSetIterator.NO_MORE_DOCS || docId + 1 == DocIdSetIterator.NO_MORE_DOCS) {
                break;
            }
            log.debug("Docs in filtered docs id set is : {}", docId);
            filteredIds[filteredIdsIndex] = docId;
            filteredIdsIndex++;
            docId++;
        }
        return filteredIds;
    }

    private Map<Integer, Float> doANNSearch(final LeafReaderContext context, final int[] filterIdsArray) throws IOException {
        SegmentReader reader = (SegmentReader) FilterLeafReader.unwrap(context.reader());
        String directory = ((FSDirectory) FilterDirectory.unwrap(reader.directory())).getDirectory().toString();

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
            if (modelMetadata == null) {
                throw new RuntimeException("Model \"" + modelId + "\" does not exist.");
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
                    getParametersAtLoading(spaceType, knnEngine, knnQuery.getIndexName()),
                    knnQuery.getIndexName()
                ),
                true
            );
        } catch (ExecutionException e) {
            GRAPH_QUERY_ERRORS.increment();
            throw new RuntimeException(e);
        }

        // Now that we have the allocation, we need to readLock it
        indexAllocation.readLock();
        try {
            if (indexAllocation.isClosed()) {
                throw new RuntimeException("Index has already been closed");
            }

            results = JNIService.queryIndex(
                indexAllocation.getMemoryAddress(),
                knnQuery.getQueryVector(),
                knnQuery.getK(),
                knnEngine.getName(),
                filterIdsArray
            );

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

    private Map<Integer, Float> doExactSearch(final LeafReaderContext leafReaderContext, final int[] filterIdsArray) {
        final SegmentReader reader = (SegmentReader) FilterLeafReader.unwrap(leafReaderContext.reader());
        final FieldInfo fieldInfo = reader.getFieldInfos().fieldInfo(knnQuery.getField());
        float[] queryVector = this.knnQuery.getQueryVector();
        try {
            final BinaryDocValues values = DocValues.getBinary(leafReaderContext.reader(), fieldInfo.name);
            final SpaceType spaceType = SpaceType.getSpace(fieldInfo.getAttribute(SPACE_TYPE));

            final Map<Integer, Float> docToScore = new HashMap<>();
            for (int j : filterIdsArray) {
                int docId = values.advance(j);
                BytesRef value = values.binaryValue();
                ByteArrayInputStream byteStream = new ByteArrayInputStream(value.bytes, value.offset, value.length);
                final KNNVectorSerializer vectorSerializer = KNNVectorSerializerFactory.getSerializerByStreamContent(byteStream);
                final float[] vector = vectorSerializer.byteToFloatArray(byteStream);
                // making min score as high score as this is closest to the vector
                float score = spaceType.getVectorSimilarityFunction().compare(queryVector, vector);
                docToScore.put(docId, score);
            }
            return docToScore;
        } catch (Exception e) {
            log.error("Error while getting the doc values to do the k-NN Search for query : {}", this.knnQuery);
        }
        return Collections.emptyMap();
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
}
