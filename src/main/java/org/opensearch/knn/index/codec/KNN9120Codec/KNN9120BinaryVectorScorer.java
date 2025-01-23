/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN9120Codec;

import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.Bits;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.KnnVectorValues;
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
        KnnVectorValues randomAccessVectorValues
    ) throws IOException {
        if (randomAccessVectorValues instanceof ByteVectorValues) {
            return new BinaryRandomVectorScorerSupplier((ByteVectorValues) randomAccessVectorValues);
        }
        throw new IllegalArgumentException("vectorValues must be an instance of RandomAccessVectorValues.Bytes");
    }

    @Override
    public RandomVectorScorer getRandomVectorScorer(
        VectorSimilarityFunction vectorSimilarityFunction,
        KnnVectorValues randomAccessVectorValues,
        float[] queryVector
    ) throws IOException {
        throw new IllegalArgumentException("binary vectors do not support float[] targets");
    }

    @Override
    public RandomVectorScorer getRandomVectorScorer(
        VectorSimilarityFunction vectorSimilarityFunction,
        KnnVectorValues randomAccessVectorValues,
        byte[] queryVector
    ) throws IOException {
        if (randomAccessVectorValues instanceof ByteVectorValues) {
            return new BinaryRandomVectorScorer((ByteVectorValues) randomAccessVectorValues, queryVector);
        }
        throw new IllegalArgumentException("vectorValues must be an instance of RandomAccessVectorValues.Bytes");
    }

    static class BinaryRandomVectorScorer implements RandomVectorScorer {
        private final ByteVectorValues vectorValues;
        private final byte[] queryVector;

        BinaryRandomVectorScorer(ByteVectorValues vectorValues, byte[] query) {
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
        protected final ByteVectorValues vectorValues;
        protected final ByteVectorValues vectorValues1;
        protected final ByteVectorValues vectorValues2;

        public BinaryRandomVectorScorerSupplier(ByteVectorValues vectorValues) throws IOException {
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
