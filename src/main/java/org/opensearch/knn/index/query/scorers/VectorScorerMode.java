/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.scorers;

import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.search.VectorScorer;

import java.io.IOException;

/**
 * Strategy for creating a {@link VectorScorer} from Lucene vector values.
 *
 * <p>This interface abstracts the choice between primary scoring and rescoring so that
 * {@link VectorScorers} can delegate scorer creation without knowing which mode is
 * in effect.
 *
 * <h2>Provided Implementations</h2>
 * <ul>
 *   <li>{@link #SCORE} — creates a scorer via {@code vectorValues.scorer(target)},
 *       which computes the primary similarity score (e.g. dot product, L2).</li>
 *   <li>{@link #RESCORE} — creates a scorer via {@code vectorValues.rescorer(target)},
 *       which recomputes a higher-fidelity score from the original (unquantized) vectors.
 *       This is typically used after an initial approximate search with quantized vectors.</li>
 * </ul>
 *
 * @see VectorScorers
 */
public interface VectorScorerMode {

    /** Creates a scorer using the primary similarity function. */
    VectorScorerMode SCORE = new VectorScorerMode() {
        @Override
        public VectorScorer createScorer(FloatVectorValues vectorValues, float[] target) throws IOException {
            return vectorValues.scorer(target);
        }

        @Override
        public VectorScorer createScorer(ByteVectorValues vectorValues, byte[] target) throws IOException {
            return vectorValues.scorer(target);
        }
    };

    /** Creates a scorer that recomputes a higher-fidelity score from unquantized vectors. */
    VectorScorerMode RESCORE = new VectorScorerMode() {
        @Override
        public VectorScorer createScorer(FloatVectorValues vectorValues, float[] target) throws IOException {
            return vectorValues.rescorer(target);
        }

        @Override
        public VectorScorer createScorer(ByteVectorValues vectorValues, byte[] target) throws IOException {
            return vectorValues.rescorer(target);
        }
    };

    /**
     * Creates a {@link VectorScorer} for a float query vector against {@link FloatVectorValues}.
     *
     * @param vectorValues the float vector values for the segment
     * @param target       the float query vector
     * @return a scorer positioned over the given vector values
     * @throws IOException if an I/O error occurs
     */
    VectorScorer createScorer(FloatVectorValues vectorValues, float[] target) throws IOException;

    /**
     * Creates a {@link VectorScorer} for a byte query vector against {@link ByteVectorValues}.
     *
     * @param vectorValues the byte vector values for the segment
     * @param target       the byte query vector
     * @return a scorer positioned over the given vector values
     * @throws IOException if an I/O error occurs
     */
    VectorScorer createScorer(ByteVectorValues vectorValues, byte[] target) throws IOException;
}
