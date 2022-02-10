/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import com.google.common.collect.ImmutableMap;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.jni.JNIService;
import org.opensearch.knn.index.memory.NativeMemoryAllocation;
import org.opensearch.knn.index.memory.NativeMemoryCacheManager;
import org.opensearch.knn.index.memory.NativeMemoryEntryContext;
import org.opensearch.knn.index.memory.NativeMemoryLoadStrategy;
import org.opensearch.knn.index.util.KNNEngine;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FilterLeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.index.Term;
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

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
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
public class KNNWeight extends Weight {
    private static Logger logger = LogManager.getLogger(KNNWeight.class);
    private static ModelDao modelDao;

    private final KNNQuery knnQuery;
    private final float boost;

    private NativeMemoryCacheManager nativeMemoryCacheManager;

    public KNNWeight(KNNQuery query, float boost) {
        super(query);
        this.knnQuery = query;
        this.boost = boost;
        this.nativeMemoryCacheManager = NativeMemoryCacheManager.getInstance();
    }

    public static void initialize(ModelDao modelDao) {
        KNNWeight.modelDao = modelDao;
    }

    @Override
    public Explanation explain(LeafReaderContext context, int doc) {
        return Explanation.match(1.0f, "No Explanation");
    }

    @Override
    public void extractTerms(Set<Term> terms) {
    }

    @Override
    public Scorer scorer(LeafReaderContext context) throws IOException {
            SegmentReader reader = (SegmentReader) FilterLeafReader.unwrap(context.reader());
            String directory = ((FSDirectory) FilterDirectory.unwrap(reader.directory())).getDirectory().toString();

            FieldInfo fieldInfo = reader.getFieldInfos().fieldInfo(knnQuery.getField());

            if (fieldInfo == null) {
                logger.debug("[KNN] Field info not found for {}:{}", knnQuery.getField(),
                        reader.getSegmentName());
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
                    ? knnEngine.getExtension() + KNNConstants.COMPOUND_EXTENSION : knnEngine.getExtension();
            String engineSuffix = knnQuery.getField() + engineExtension;
            List<String> engineFiles = reader.getSegmentInfo().files().stream()
                    .filter(fileName -> fileName.endsWith(engineSuffix))
                    .collect(Collectors.toList());

            if(engineFiles.isEmpty()) {
                logger.debug("[KNN] No engine index found for field {} for segment {}",
                        knnQuery.getField(), reader.getSegmentName());
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
                        ), true);
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

                results = JNIService.queryIndex(indexAllocation.getMemoryAddress(), knnQuery.getQueryVector(), knnQuery.getK(), knnEngine.getName());
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
                logger.debug("[KNN] Query yielded 0 results");
                return null;
            }

            Map<Integer, Float> scores = Arrays.stream(results).collect(
                    Collectors.toMap(KNNQueryResult::getId, result -> knnEngine.score(result.getScore(), spaceType)));
            int maxDoc = Collections.max(scores.keySet()) + 1;
            DocIdSetBuilder docIdSetBuilder = new DocIdSetBuilder(maxDoc);
            DocIdSetBuilder.BulkAdder setAdder = docIdSetBuilder.grow(maxDoc);
            Arrays.stream(results).forEach(result -> setAdder.add(result.getId()));
            DocIdSetIterator docIdSetIter = docIdSetBuilder.build().iterator();
            return new KNNScorer(this, docIdSetIter, scores, boost);
    }

    @Override
    public boolean isCacheable(LeafReaderContext context) {
        return true;
    }

    public static float normalizeScore(float score) {
        if (score >= 0)
            return 1 / (1 + score);
        return -score + 1;
    }
}

