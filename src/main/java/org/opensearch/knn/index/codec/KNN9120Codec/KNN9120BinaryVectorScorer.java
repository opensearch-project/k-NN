/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN9120Codec;

import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;
import org.apache.lucene.util.hnsw.UpdateableRandomVectorScorer;
import org.opensearch.knn.index.KNNVectorSimilarityFunction;

import java.io.IOException;

/**
 * A {@link FlatVectorsScorer} for scoring binary (Hamming) vectors. Used by {@link KNN9120HnswBinaryVectorsFormat}.
 * The query path returns a fixed-target {@link BinaryRandomVectorScorer}; graph construction uses the updateable
 * {@link BinaryUpdatableRandomVectorScorer} produced by {@link BinaryRandomVectorScorerSupplier}.
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
            // The query path uses a fixed-target scorer that extends AbstractRandomVectorScorer, mirroring Lucene's
            // DefaultFlatVectorScorer. Only the supplier path (graph construction) needs the updateable scorer below.
            return new BinaryRandomVectorScorer((ByteVectorValues) randomAccessVectorValues, queryVector);
        }
        throw new IllegalArgumentException("vectorValues must be an instance of RandomAccessVectorValues.Bytes");
    }

    /**
     * Fixed-target query scorer for binary vectors. Mirrors Lucene's {@code DefaultFlatVectorScorer}, which returns a
     * plain {@link RandomVectorScorer.AbstractRandomVectorScorer} for the query path (as opposed to the updateable
     * {@link BinaryUpdatableRandomVectorScorer} used during graph construction). Extending {@code AbstractRandomVectorScorer}
     * keeps the query scorer compatible with wrappers such as the prefetching scorer.
     */
    static class BinaryRandomVectorScorer extends RandomVectorScorer.AbstractRandomVectorScorer {
        private final byte[] queryVector;

        BinaryRandomVectorScorer(ByteVectorValues vectorValues, byte[] query) {
            super(vectorValues);
            this.queryVector = query;
        }

        @Override
        public float score(int node) throws IOException {
            return KNNVectorSimilarityFunction.HAMMING.compare(queryVector, ((ByteVectorValues) values()).vectorValue(node));
        }
    }

    /**
     * Just like a {@link BinaryRandomVectorScorer} but allows the scoring ordinal to be changed. Useful
     * during indexing operations.
     *
     * <p>Mirrors Lucene's {@code DefaultFlatVectorScorer}: reads vectors for both {@link #score(int)} and
     * {@link #setScoringOrdinal(int)} through an independent {@code targetVectors} copy (a distinct
     * {@link ByteVectorValues} with its own {@code IndexInput} cursor and read buffer), while the base
     * {@code values()} is used only for graph metadata. Using a separate copy avoids sharing a stateful
     * cursor/buffer with the base values during graph construction.
     */
    static class BinaryUpdatableRandomVectorScorer extends UpdateableRandomVectorScorer.AbstractUpdateableRandomVectorScorer {
        private final ByteVectorValues targetVectors;
        private final byte[] vector;

        BinaryUpdatableRandomVectorScorer(ByteVectorValues vectorValues, ByteVectorValues targetVectors, byte[] vector) {
            super(vectorValues);
            this.targetVectors = targetVectors;
            this.vector = vector;
        }

        @Override
        public float score(int node) throws IOException {
            return KNNVectorSimilarityFunction.HAMMING.compare(vector, targetVectors.vectorValue(node));
        }

        @Override
        public void setScoringOrdinal(int node) throws IOException {
            System.arraycopy(targetVectors.vectorValue(node), 0, vector, 0, vector.length);
        }
    }

    /**
     * A supplier that creates {@link RandomVectorScorer} from an ordinal.
     */
    static class BinaryRandomVectorScorerSupplier implements RandomVectorScorerSupplier {
        protected final ByteVectorValues vectorValues;
        protected final ByteVectorValues targetVectors;

        public BinaryRandomVectorScorerSupplier(ByteVectorValues vectorValues) throws IOException {
            this.vectorValues = vectorValues;
            this.targetVectors = vectorValues.copy();
        }

        @Override
        public UpdateableRandomVectorScorer scorer() throws IOException {
            byte[] query = new byte[vectorValues.dimension()];
            return new BinaryUpdatableRandomVectorScorer(vectorValues, targetVectors, query);
        }

        @Override
        public RandomVectorScorerSupplier copy() throws IOException {
            return new BinaryRandomVectorScorerSupplier(vectorValues.copy());
        }
    }
}
