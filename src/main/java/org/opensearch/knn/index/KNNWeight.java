/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */
/*
 *   Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 *   Licensed under the Apache License, Version 2.0 (the "License").
 *   You may not use this file except in compliance with the License.
 *   A copy of the License is located at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   or in the "license" file accompanying this file. This file is distributed
 *   on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 *   express or implied. See the License for the specific language governing
 *   permissions and limitations under the License.
 */

package org.opensearch.knn.index;

import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.codec.KNNCodecUtil;
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
import org.opensearch.knn.plugin.stats.KNNCounter;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;

/**
 * Calculate query weights and build query scorers.
 */
public class KNNWeight extends Weight {
    private static Logger logger = LogManager.getLogger(KNNWeight.class);
    private final KNNQuery knnQuery;
    private final float boost;

    public static KNNIndexCache knnIndexCache = KNNIndexCache.getInstance();

    public KNNWeight(KNNQuery query, float boost) {
        super(query);
        this.knnQuery = query;
        this.boost = boost;
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

            KNNEngine knnEngine = KNNEngine.getEngine(fieldInfo.getAttribute(KNN_ENGINE));
            SpaceType spaceType = SpaceType.getSpace(fieldInfo.getAttribute(KNNConstants.SPACE_TYPE));

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

            try {
                results = knnIndexCache.queryIndex(indexPath.toString(), knnQuery.getIndexName(),
                        spaceType, knnQuery.getQueryVector(), knnQuery.getK(), knnEngine.getName());
            } catch (Exception ex) {
                KNNCounter.GRAPH_QUERY_ERRORS.increment();
                throw ex;
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

