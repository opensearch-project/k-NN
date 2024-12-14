/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN9120Codec;

import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.hnsw.RandomAccessVectorValues;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;
import org.opensearch.knn.index.KNNVectorSimilarityFunction;

import java.io.IOException;

/**
 * A FlatVectorsScorer to be used for scoring binary vectors. Meant to be used with {@link KNN9120BinaryVectorScorer}
 */
public class KNN9120BinaryVectorScorer implements FlatVectorsScorer {
    @Override
    public RandomVectorScorerSupplier getRandomVectorScorerSupplier(
        VectorSimilarityFunction vectorSimilarityFunction,
        RandomAccessVectorValues randomAccessVectorValues
    ) throws IOException {
        if (randomAccessVectorValues instanceof RandomAccessVectorValues.Bytes) {
            return new BinaryRandomVectorScorerSupplier((RandomAccessVectorValues.Bytes) randomAccessVectorValues);
        }
        throw new IllegalArgumentException("vectorValues must be an instance of RandomAccessVectorValues.Bytes");
    }

    @Override
    public RandomVectorScorer getRandomVectorScorer(
        VectorSimilarityFunction vectorSimilarityFunction,
        RandomAccessVectorValues randomAccessVectorValues,
        float[] queryVector
    ) throws IOException {
        throw new IllegalArgumentException("binary vectors do not support float[] targets");
    }

    @Override
    public RandomVectorScorer getRandomVectorScorer(
        VectorSimilarityFunction vectorSimilarityFunction,
        RandomAccessVectorValues randomAccessVectorValues,
        byte[] queryVector
    ) throws IOException {
        if (randomAccessVectorValues instanceof RandomAccessVectorValues.Bytes) {
            return new BinaryRandomVectorScorer((RandomAccessVectorValues.Bytes) randomAccessVectorValues, queryVector);
        }
        throw new IllegalArgumentException("vectorValues must be an instance of RandomAccessVectorValues.Bytes");
    }

    static class BinaryRandomVectorScorer implements RandomVectorScorer {
        private final RandomAccessVectorValues.Bytes vectorValues;
        private final byte[] queryVector;

        BinaryRandomVectorScorer(RandomAccessVectorValues.Bytes vectorValues, byte[] query) {
            this.queryVector = query;
            this.vectorValues = vectorValues;
        }

        @Override
        public float score(int node) throws IOException {
            return KNNVectorSimilarityFunction.HAMMING.compare(queryVector, vectorValues.vectorValue(node));
        }

        @Override
        public int maxOrd() {
            return vectorValues.size();
        }

        @Override
        public int ordToDoc(int ord) {
            return vectorValues.ordToDoc(ord);
        }

        @Override
        public Bits getAcceptOrds(Bits acceptDocs) {
            return vectorValues.getAcceptOrds(acceptDocs);
        }
    }

    static class BinaryRandomVectorScorerSupplier implements RandomVectorScorerSupplier {
        protected final RandomAccessVectorValues.Bytes vectorValues;
        protected final RandomAccessVectorValues.Bytes vectorValues1;
        protected final RandomAccessVectorValues.Bytes vectorValues2;

        public BinaryRandomVectorScorerSupplier(RandomAccessVectorValues.Bytes vectorValues) throws IOException {
            this.vectorValues = vectorValues;
            this.vectorValues1 = vectorValues.copy();
            this.vectorValues2 = vectorValues.copy();
        }

        @Override
        public RandomVectorScorer scorer(int ord) throws IOException {
            byte[] queryVector = vectorValues1.vectorValue(ord);
            return new BinaryRandomVectorScorer(vectorValues2, queryVector);
        }

        @Override
        public RandomVectorScorerSupplier copy() throws IOException {
            return new BinaryRandomVectorScorerSupplier(vectorValues.copy());
        }
    }
}
