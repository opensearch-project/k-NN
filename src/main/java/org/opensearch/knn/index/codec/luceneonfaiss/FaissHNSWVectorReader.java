/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.luceneonfaiss;

import org.apache.lucene.codecs.hnsw.FlatVectorScorerUtil;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.IOSupplier;
import org.apache.lucene.util.hnsw.HnswGraphSearcher;
import org.apache.lucene.util.hnsw.OrdinalTranslatedKnnCollector;
import org.apache.lucene.util.hnsw.RandomVectorScorer;

import java.io.Closeable;
import java.io.IOException;

public class FaissHNSWVectorReader implements Closeable {
    private static FlatVectorsScorer VECTOR_SCORER = FlatVectorScorerUtil.getLucene99FlatVectorsScorer();

    private IndexInput indexInput;
    private FaissIdMapIndex faissIdMapIndex;
    private FaissIndexFlat faissIndexFlat;
    private LuceneFaissHnswGraph faissHnswGraph;

    public FaissHNSWVectorReader(IndexInput indexInput) throws IOException {
        this.indexInput = indexInput;
        faissIdMapIndex = (FaissIdMapIndex) FaissIndex.load(indexInput);
        final FaissHNSWFlatIndex faissHNSWFlatIndex = faissIdMapIndex.getNestedIndex();
        faissIndexFlat = faissHNSWFlatIndex.getStorage();
        faissHnswGraph = new LuceneFaissHnswGraph(faissIdMapIndex.getNestedIndex(), indexInput);
    }

    public void search(float[] target, KnnCollector knnCollector, Bits acceptDocs) throws IOException {
        search(VectorEncoding.FLOAT32,
               () -> VECTOR_SCORER.getRandomVectorScorer(faissIndexFlat.getSimilarityFunction(),
                                                         faissIdMapIndex.getFloatValues(indexInput),
                                                         target
               ),
               knnCollector,
               acceptDocs
        );
    }

    private void search(
        final VectorEncoding vectorEncoding,
        final IOSupplier<RandomVectorScorer> scorerSupplier,
        final KnnCollector knnCollector,
        final Bits acceptDocs
    ) throws IOException {
        if (faissIndexFlat.getTotalNumberOfVectors() == 0 || knnCollector.k() == 0
            || faissIndexFlat.getVectorEncoding() != vectorEncoding) {
            return;
        }

        final RandomVectorScorer scorer = scorerSupplier.get();
        final KnnCollector collector = new OrdinalTranslatedKnnCollector(knnCollector, scorer::ordToDoc);
        final Bits acceptedOrds = scorer.getAcceptOrds(acceptDocs);

        if (knnCollector.k() < scorer.maxOrd()) {
            HnswGraphSearcher.search(scorer, collector, faissHnswGraph, acceptedOrds);
        } else {
            // if k is larger than the number of vectors, we can just iterate over all vectors
            // and collect them.
            for (int i = 0; i < scorer.maxOrd(); i++) {
                if (acceptedOrds == null || acceptedOrds.get(i)) {
                    if (!knnCollector.earlyTerminated()) {
                        knnCollector.incVisitedCount(1);
                        knnCollector.collect(scorer.ordToDoc(i), scorer.score(i));
                    } else {
                        break;
                    }
                }
            }
        }  // End if
    }

    @Override
    public void close() throws IOException {
        indexInput.close();
    }
}
