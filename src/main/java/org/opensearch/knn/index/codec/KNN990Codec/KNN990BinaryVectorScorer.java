/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.hnsw.RandomAccessVectorValues;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;

import java.io.IOException;

public class KNN990BinaryVectorScorer implements FlatVectorsScorer {
    @Override
    public RandomVectorScorerSupplier getRandomVectorScorerSupplier(
        VectorSimilarityFunction vectorSimilarityFunction,
        RandomAccessVectorValues randomAccessVectorValues
    ) throws IOException {
        assert randomAccessVectorValues instanceof RandomAccessVectorValues.Bytes;
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
        assert randomAccessVectorValues instanceof RandomAccessVectorValues.Bytes;
        if (randomAccessVectorValues instanceof RandomAccessVectorValues.Bytes) {
            return new BinaryRandomVectorScorer((RandomAccessVectorValues.Bytes) randomAccessVectorValues, queryVector);
        }
        throw new IllegalArgumentException("vectorValues must be an instance of RandomAccessVectorValues.Bytes");
    }

    static class BinaryRandomVectorScorer implements RandomVectorScorer {
        private final RandomAccessVectorValues.Bytes vectorValues;
        private final int bitDimensions;
        private final byte[] queryVector;

        BinaryRandomVectorScorer(RandomAccessVectorValues.Bytes vectorValues, byte[] query) {
            this.queryVector = query;
            this.bitDimensions = vectorValues.dimension() * Byte.SIZE;
            this.vectorValues = vectorValues;
        }

        @Override
        public float score(int node) throws IOException {
            return (bitDimensions - VectorUtil.xorBitCount(queryVector, vectorValues.vectorValue(node))) / (float) bitDimensions;
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
