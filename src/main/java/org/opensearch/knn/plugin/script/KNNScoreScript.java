/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.script;

import org.opensearch.knn.index.KNNVectorScriptDocValues;
import org.apache.lucene.index.LeafReaderContext;
import org.opensearch.index.fielddata.ScriptDocValues;
import org.opensearch.script.ScoreScript;
import org.opensearch.search.lookup.SearchLookup;

import java.io.IOException;
import java.math.BigInteger;
import java.util.Map;
import java.util.function.BiFunction;

/**
 * KNNScoreScript is used for adjusting the score of query results based on similarity distance methods. Scripts
 * operate on a per document basis. Because the scoring method is passed in during construction, KNNScoreScripts are
 * only concerned with the types of the query and docs being processed.
 */
public abstract class KNNScoreScript<T> extends ScoreScript {
    protected final T queryValue;
    protected final String field;
    protected final BiFunction<T, T, Float> scoringMethod;

    public KNNScoreScript(
        Map<String, Object> params,
        T queryValue,
        String field,
        BiFunction<T, T, Float> scoringMethod,
        SearchLookup lookup,
        LeafReaderContext leafContext
    ) {
        super(params, lookup, leafContext);
        this.queryValue = queryValue;
        this.field = field;
        this.scoringMethod = scoringMethod;
    }

    /**
     * KNNScoreScript with Long type. The query value passed in as well as the DocValues being searched over are
     * expected to be Longs.
     */
    public static class LongType extends KNNScoreScript<Long> {
        public LongType(
            Map<String, Object> params,
            Long queryValue,
            String field,
            BiFunction<Long, Long, Float> scoringMethod,
            SearchLookup lookup,
            LeafReaderContext leafContext
        ) {
            super(params, queryValue, field, scoringMethod, lookup, leafContext);
        }

        /**
         * This function calculates the similarity score for each doc in the segment.
         *
         * @param explanationHolder A helper to take in an explanation from a script and turn
         *                          it into an {@link org.apache.lucene.search.Explanation}
         * @return score for the provided space between the doc and the query
         */
        @Override
        public double execute(ScoreScript.ExplanationHolder explanationHolder) {
            ScriptDocValues.Longs scriptDocValues = (ScriptDocValues.Longs) getDoc().get(this.field);
            if (scriptDocValues.isEmpty()) {
                return 0.0;
            }
            return this.scoringMethod.apply(this.queryValue, scriptDocValues.getValue());
        }
    }

    /**
     * KNNScoreScript with BigInteger type. The query value passed in as well as the DocValues being searched over
     * are expected to be BigInteger.
     */
    public static class BigIntegerType extends KNNScoreScript<BigInteger> {
        public BigIntegerType(
            Map<String, Object> params,
            BigInteger queryValue,
            String field,
            BiFunction<BigInteger, BigInteger, Float> scoringMethod,
            SearchLookup lookup,
            LeafReaderContext leafContext
        ) {
            super(params, queryValue, field, scoringMethod, lookup, leafContext);
        }

        /**
         * This function calculates the similarity score for each doc in the segment.
         *
         * @param explanationHolder A helper to take in an explanation from a script and turn
         *                          it into an {@link org.apache.lucene.search.Explanation}
         * @return score for the provided space between the doc and the query
         */
        @Override
        public double execute(ScoreScript.ExplanationHolder explanationHolder) {
            ScriptDocValues.BytesRefs scriptDocValues = (ScriptDocValues.BytesRefs) getDoc().get(this.field);
            if (scriptDocValues.isEmpty()) {
                return 0.0;
            }
            return this.scoringMethod.apply(this.queryValue, new BigInteger(1, scriptDocValues.getValue().bytes));
        }
    }

    /**
     * KNNVectors with float[] type. The query value passed in is expected to be float[]. The fieldType of the docs
     * being searched over are expected to be KNNVector type.
     */
    public static class KNNVectorType extends KNNScoreScript<float[]> {

        public KNNVectorType(
            Map<String, Object> params,
            float[] queryValue,
            String field,
            BiFunction<float[], float[], Float> scoringMethod,
            SearchLookup lookup,
            LeafReaderContext leafContext
        ) throws IOException {
            super(params, queryValue, field, scoringMethod, lookup, leafContext);
        }

        /**
         * This function called for each doc in the segment. We evaluate the score of the vector in the doc
         *
         * @param explanationHolder A helper to take in an explanation from a script and turn
         *                          it into an {@link org.apache.lucene.search.Explanation}
         * @return score of the vector to the query vector
         */
        @Override
        public double execute(ScoreScript.ExplanationHolder explanationHolder) {
            KNNVectorScriptDocValues scriptDocValues = (KNNVectorScriptDocValues) getDoc().get(this.field);
            if (scriptDocValues.isEmpty()) {
                return 0.0;
            }
            return this.scoringMethod.apply(this.queryValue, scriptDocValues.getValue());
        }
    }
}
