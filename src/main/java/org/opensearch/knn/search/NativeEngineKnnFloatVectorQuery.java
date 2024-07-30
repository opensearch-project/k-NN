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

package org.opensearch.knn.search;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.QueryTimeout;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.BitSetIterator;
import org.apache.lucene.util.Bits;
import org.opensearch.knn.index.query.KNNQuery;
import org.opensearch.knn.index.query.KNNWeight;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

public class NativeEngineKnnFloatVectorQuery extends KnnFloatVectorQuery {

    private final KNNQuery query;
    private final KNNWeight knnWeight;

    public NativeEngineKnnFloatVectorQuery(final KNNQuery knnQuery) {
        super(knnQuery.getField(), knnQuery.getQueryVector(), knnQuery.getK(), knnQuery.getFilterQuery());
        this.query = knnQuery;
        knnWeight = new KNNWeight(query, 1);
    }

    protected TopDocs approximateSearch(
        LeafReaderContext context,
        Bits acceptDocs,
        int visitedLimit,
        KnnCollectorManager knnCollectorManager
    ) throws IOException {

        BitSet filterBitset = null;
        int cardinality = 0;

        if (query.getFilterQuery() != null && acceptDocs != null) {
            // If filter query is null and accepted doc are not null then we can directly convert it to bitset
            // This maintains the status quo with current code path wherein deleted docs are not considered
            // Without this check deleted docs will be passed in and impact latencies.
            filterBitset = (BitSet) acceptDocs;
            cardinality = filterBitset.cardinality();
        }

        Map<Integer, Float> leafDocScores = knnWeight.doANNSearch(context, filterBitset, cardinality);
        if (leafDocScores == null) {
            leafDocScores = Collections.emptyMap();
        }
        final Bits liveDocs = context.reader().getLiveDocs();

        final List<Map.Entry<Integer, Float>> topScores = new ArrayList<>(leafDocScores.entrySet());
        topScores.sort(Map.Entry.<Integer, Float>comparingByValue().reversed());

        //This is to trick the implementation to force exactsearch after approxsearch
        TotalHits.Relation relation = query.getFilterQuery() != null
                && cardinality >= query.getK() && topScores.size() < query.getK()
                ? TotalHits.Relation.GREATER_THAN_OR_EQUAL_TO
                : TotalHits.Relation.EQUAL_TO;

        return convertDocScoresToTopDocs(topScores, liveDocs, context.docBase, relation);
    }

    protected TopDocs exactSearch(LeafReaderContext context, DocIdSetIterator acceptIterator, QueryTimeout queryTimeout) {
        if (acceptIterator instanceof BitSetIterator) {
            BitSetIterator bitSetIterator = (BitSetIterator) acceptIterator;
            BitSet acceptedBitSet = bitSetIterator.getBitSet();

            final Map<Integer, Float> docToScore = knnWeight.doExactSearch(context, acceptedBitSet, acceptedBitSet.cardinality());
            return convertDocScoresToTopDocs(new ArrayList<>(docToScore.entrySet()), null, context.docBase, TotalHits.Relation.EQUAL_TO);
        }
        throw new IllegalStateException("DocIdSetIterator is not a BitSetIterator");
    }

    private TopDocs convertDocScoresToTopDocs(final List<Map.Entry<Integer, Float>> docScores, final Bits liveDocs, int docBase, TotalHits.Relation relation) {
        final List<ScoreDoc> scoreDocs = new ArrayList<>(docScores.size());
        int totalHits = 0;
        for (final Map.Entry<Integer, Float> entry : docScores) {
            // since lucene query executes at shard, we need to filter delete docs
            if (liveDocs == null || liveDocs.get(entry.getKey())) {
                ScoreDoc scoreDoc = new ScoreDoc(entry.getKey() + docBase, entry.getValue());
                scoreDocs.add(scoreDoc);
                totalHits++;
            }
        }

        return new TopDocs(new TotalHits(totalHits, relation), scoreDocs.toArray(ScoreDoc[]::new));
    }

    @Override
    protected TopDocs mergeLeafResults(TopDocs[] perLeafResults) {
        // Rescore here
        return TopDocs.merge(k, perLeafResults);
    }
}
