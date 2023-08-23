/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.script;

import org.apache.lucene.search.IndexSearcher;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.index.query.KNNWeight;
import org.apache.lucene.index.LeafReaderContext;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.script.ScoreScript;
import org.opensearch.search.lookup.SearchLookup;

import java.io.IOException;
import java.math.BigInteger;
import java.util.Map;
import java.util.function.BiFunction;

import static org.opensearch.knn.plugin.script.KNNScoringSpaceUtil.getVectorMagnitudeSquared;
import static org.opensearch.knn.plugin.script.KNNScoringSpaceUtil.isBinaryFieldType;
import static org.opensearch.knn.plugin.script.KNNScoringSpaceUtil.isKNNVectorFieldType;
import static org.opensearch.knn.plugin.script.KNNScoringSpaceUtil.isLongFieldType;
import static org.opensearch.knn.plugin.script.KNNScoringSpaceUtil.parseToBigInteger;
import static org.opensearch.knn.plugin.script.KNNScoringSpaceUtil.parseToFloatArray;
import static org.opensearch.knn.plugin.script.KNNScoringSpaceUtil.parseToLong;

public interface KNNScoringSpace {
    /**
     * Return the correct scoring script for a given query. The scoring script
     *
     * @param params   Map of parameters
     * @param field    Fieldname
     * @param lookup   SearchLookup
     * @param ctx      ctx LeafReaderContext to be used for scoring documents
     * @param searcher IndexSearcher
     * @return ScoreScript for this query
     * @throws IOException throws IOException if ScoreScript cannot be constructed
     */
    ScoreScript getScoreScript(Map<String, Object> params, String field, SearchLookup lookup, LeafReaderContext ctx, IndexSearcher searcher)
        throws IOException;

    class L2 implements KNNScoringSpace {

        float[] processedQuery;
        BiFunction<float[], float[], Float> scoringMethod;

        /**
         * Constructor for L2 scoring space. L2 scoring space expects values to be of type float[].
         *
         * @param query Query object that, along with the doc values, will be used to compute L2 score
         * @param fieldType FieldType for the doc values that will be used
         */
        public L2(Object query, MappedFieldType fieldType) {
            if (!isKNNVectorFieldType(fieldType)) {
                throw new IllegalArgumentException("Incompatible field_type for l2 space. The field type must " + "be knn_vector.");
            }

            this.processedQuery = parseToFloatArray(
                query,
                ((KNNVectorFieldMapper.KNNVectorFieldType) fieldType).getDimension(),
                ((KNNVectorFieldMapper.KNNVectorFieldType) fieldType).getVectorDataType()
            );
            this.scoringMethod = (float[] q, float[] v) -> 1 / (1 + KNNScoringUtil.l2Squared(q, v));
        }

        public ScoreScript getScoreScript(
            Map<String, Object> params,
            String field,
            SearchLookup lookup,
            LeafReaderContext ctx,
            IndexSearcher searcher
        ) throws IOException {
            return new KNNScoreScript.KNNVectorType(params, this.processedQuery, field, this.scoringMethod, lookup, ctx, searcher);
        }
    }

    class CosineSimilarity implements KNNScoringSpace {

        float[] processedQuery;
        BiFunction<float[], float[], Float> scoringMethod;

        /**
         * Constructor for CosineSimilarity scoring space. CosineSimilarity scoring space expects values to be of type
         * float[].
         *
         * @param query Query object that, along with the doc values, will be used to compute CosineSimilarity score
         * @param fieldType FieldType for the doc values that will be used
         */
        public CosineSimilarity(Object query, MappedFieldType fieldType) {
            if (!isKNNVectorFieldType(fieldType)) {
                throw new IllegalArgumentException("Incompatible field_type for cosine space. The field type must " + "be knn_vector.");
            }

            this.processedQuery = parseToFloatArray(
                query,
                ((KNNVectorFieldMapper.KNNVectorFieldType) fieldType).getDimension(),
                ((KNNVectorFieldMapper.KNNVectorFieldType) fieldType).getVectorDataType()
            );
            float qVectorSquaredMagnitude = getVectorMagnitudeSquared(this.processedQuery);
            this.scoringMethod = (float[] q, float[] v) -> 1 + KNNScoringUtil.cosinesimilOptimized(q, v, qVectorSquaredMagnitude);
        }

        public ScoreScript getScoreScript(
            Map<String, Object> params,
            String field,
            SearchLookup lookup,
            LeafReaderContext ctx,
            IndexSearcher searcher
        ) throws IOException {
            return new KNNScoreScript.KNNVectorType(params, this.processedQuery, field, this.scoringMethod, lookup, ctx, searcher);
        }
    }

    class HammingBit implements KNNScoringSpace {

        Object processedQuery;
        BiFunction<?, ?, Float> scoringMethod;

        /**
         * Constructor for HammingBit scoring space. HammingBit scoring space expects values to either be of type
         * Long or Base64 encoded strings.
         *
         * @param query Query object that, along with the doc values, will be used to compute HammingBit score
         * @param fieldType FieldType for the doc values that will be used
         */
        public HammingBit(Object query, MappedFieldType fieldType) {
            if (isLongFieldType(fieldType)) {
                this.processedQuery = parseToLong(query);
                this.scoringMethod = (Long q, Long v) -> 1.0f / (1 + KNNScoringUtil.calculateHammingBit(q, v));
            } else if (isBinaryFieldType(fieldType)) {
                this.processedQuery = parseToBigInteger(query);
                this.scoringMethod = (BigInteger q, BigInteger v) -> 1.0f / (1 + KNNScoringUtil.calculateHammingBit(q, v));
            } else {
                throw new IllegalArgumentException(
                    "Incompatible field_type for hamming space. The field type must " + "of type long or binary."
                );
            }
        }

        @SuppressWarnings("unchecked")
        public ScoreScript getScoreScript(
            Map<String, Object> params,
            String field,
            SearchLookup lookup,
            LeafReaderContext ctx,
            IndexSearcher searcher
        ) throws IOException {
            if (this.processedQuery instanceof Long) {
                return new KNNScoreScript.LongType(
                    params,
                    (Long) this.processedQuery,
                    field,
                    (BiFunction<Long, Long, Float>) this.scoringMethod,
                    lookup,
                    ctx,
                    searcher
                );
            }

            return new KNNScoreScript.BigIntegerType(
                params,
                (BigInteger) this.processedQuery,
                field,
                (BiFunction<BigInteger, BigInteger, Float>) this.scoringMethod,
                lookup,
                ctx,
                searcher
            );
        }
    }

    class L1 implements KNNScoringSpace {

        float[] processedQuery;
        BiFunction<float[], float[], Float> scoringMethod;

        /**
         * Constructor for L1 scoring space. L1 scoring space expects values to be of type float[].
         *
         * @param query Query object that, along with the doc values, will be used to compute L1 score
         * @param fieldType FieldType for the doc values that will be used
         */
        public L1(Object query, MappedFieldType fieldType) {
            if (!isKNNVectorFieldType(fieldType)) {
                throw new IllegalArgumentException("Incompatible field_type for l1 space. The field type must " + "be knn_vector.");
            }

            this.processedQuery = parseToFloatArray(
                query,
                ((KNNVectorFieldMapper.KNNVectorFieldType) fieldType).getDimension(),
                ((KNNVectorFieldMapper.KNNVectorFieldType) fieldType).getVectorDataType()
            );
            this.scoringMethod = (float[] q, float[] v) -> 1 / (1 + KNNScoringUtil.l1Norm(q, v));
        }

        public ScoreScript getScoreScript(
            Map<String, Object> params,
            String field,
            SearchLookup lookup,
            LeafReaderContext ctx,
            IndexSearcher searcher
        ) throws IOException {
            return new KNNScoreScript.KNNVectorType(params, this.processedQuery, field, this.scoringMethod, lookup, ctx, searcher);
        }
    }

    class LInf implements KNNScoringSpace {

        float[] processedQuery;
        BiFunction<float[], float[], Float> scoringMethod;

        /**
         * Constructor for L-inf scoring space. L-inf scoring space expects values to be of type float[].
         *
         * @param query Query object that, along with the doc values, will be used to compute L-inf score
         * @param fieldType FieldType for the doc values that will be used
         */
        public LInf(Object query, MappedFieldType fieldType) {
            if (!isKNNVectorFieldType(fieldType)) {
                throw new IllegalArgumentException("Incompatible field_type for l-inf space. The field type must " + "be knn_vector.");
            }

            this.processedQuery = parseToFloatArray(
                query,
                ((KNNVectorFieldMapper.KNNVectorFieldType) fieldType).getDimension(),
                ((KNNVectorFieldMapper.KNNVectorFieldType) fieldType).getVectorDataType()
            );
            this.scoringMethod = (float[] q, float[] v) -> 1 / (1 + KNNScoringUtil.lInfNorm(q, v));
        }

        public ScoreScript getScoreScript(
            Map<String, Object> params,
            String field,
            SearchLookup lookup,
            LeafReaderContext ctx,
            IndexSearcher searcher
        ) throws IOException {
            return new KNNScoreScript.KNNVectorType(params, this.processedQuery, field, this.scoringMethod, lookup, ctx, searcher);
        }
    }

    class InnerProd implements KNNScoringSpace {

        float[] processedQuery;
        BiFunction<float[], float[], Float> scoringMethod;

        /**
         * Constructor for innerproduct scoring space. innerproduct scoring space expects values to be of type float[].
         *
         * @param query Query object that, along with the doc values, will be used to compute L-inf score
         * @param fieldType FieldType for the doc values that will be used
         */
        public InnerProd(Object query, MappedFieldType fieldType) {
            if (!isKNNVectorFieldType(fieldType)) {
                throw new IllegalArgumentException(
                    "Incompatible field_type for innerproduct space. The field type must " + "be knn_vector."
                );
            }

            this.processedQuery = parseToFloatArray(
                query,
                ((KNNVectorFieldMapper.KNNVectorFieldType) fieldType).getDimension(),
                ((KNNVectorFieldMapper.KNNVectorFieldType) fieldType).getVectorDataType()
            );
            this.scoringMethod = (float[] q, float[] v) -> KNNWeight.normalizeScore(-KNNScoringUtil.innerProduct(q, v));
        }

        @Override
        public ScoreScript getScoreScript(
            Map<String, Object> params,
            String field,
            SearchLookup lookup,
            LeafReaderContext ctx,
            IndexSearcher searcher
        ) throws IOException {
            return new KNNScoreScript.KNNVectorType(params, this.processedQuery, field, this.scoringMethod, lookup, ctx, searcher);
        }
    }
}
