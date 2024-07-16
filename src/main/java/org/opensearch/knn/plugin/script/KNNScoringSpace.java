/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.script;

import org.apache.lucene.search.IndexSearcher;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapperUtil;
import org.opensearch.knn.index.query.KNNWeight;
import org.apache.lucene.index.LeafReaderContext;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.script.ScoreScript;
import org.opensearch.search.lookup.SearchLookup;

import java.io.IOException;
import java.math.BigInteger;
import java.util.Locale;
import java.util.Map;
import java.util.function.BiFunction;

import static org.opensearch.knn.plugin.script.KNNScoringSpaceUtil.getVectorMagnitudeSquared;
import static org.opensearch.knn.plugin.script.KNNScoringSpaceUtil.isBinaryFieldType;
import static org.opensearch.knn.plugin.script.KNNScoringSpaceUtil.isBinaryVectorDataType;
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
     * @param field    field name
     * @param lookup   SearchLookup
     * @param ctx      ctx LeafReaderContext to be used for scoring documents
     * @param searcher IndexSearcher
     * @return ScoreScript for this query
     * @throws IOException throws IOException if ScoreScript cannot be constructed
     */
    ScoreScript getScoreScript(Map<String, Object> params, String field, SearchLookup lookup, LeafReaderContext ctx, IndexSearcher searcher)
        throws IOException;

    /**
     * Base class to represent linear space
     *
     * As of now, all supporting spaces are linear space except hamming.
     * LinearSpace does not support binary vector.
     */
    abstract class LinearSpace implements KNNScoringSpace {
        float[] processedQuery;
        BiFunction<float[], float[], Float> scoringMethod;

        public LinearSpace(final Object query, final MappedFieldType fieldType, final String spaceName) {
            KNNVectorFieldMapper.KNNVectorFieldType knnVectorFieldType = toKNNVectorFieldType(fieldType, spaceName);
            this.processedQuery = getProcessedQuery(query, knnVectorFieldType);
            this.scoringMethod = getScoringMethod(this.processedQuery);
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

        private KNNVectorFieldMapper.KNNVectorFieldType toKNNVectorFieldType(final MappedFieldType fieldType, final String spaceName) {
            if (isKNNVectorFieldType(fieldType) == false) {
                throw new IllegalArgumentException(
                    String.format(Locale.ROOT, "Incompatible field_type for %s space. The field type must be knn_vector.", spaceName)
                );
            }

            KNNVectorFieldMapper.KNNVectorFieldType knnVectorFieldType = (KNNVectorFieldMapper.KNNVectorFieldType) fieldType;
            if (isBinaryVectorDataType(knnVectorFieldType)) {
                throw new IllegalArgumentException(
                    String.format(
                        Locale.ROOT,
                        "Incompatible field_type for %s space. The data type should be either float or byte but got %s",
                        spaceName,
                        knnVectorFieldType.getVectorDataType().getValue()
                    )
                );
            }

            return knnVectorFieldType;
        }

        protected float[] getProcessedQuery(final Object query, final KNNVectorFieldMapper.KNNVectorFieldType knnVectorFieldType) {
            return parseToFloatArray(
                query,
                KNNVectorFieldMapperUtil.getExpectedVectorLength(knnVectorFieldType),
                knnVectorFieldType.getVectorDataType()
            );
        }

        protected abstract BiFunction<float[], float[], Float> getScoringMethod(final float[] processedQuery);

    }

    class L2 extends LinearSpace {
        public L2(final Object query, final MappedFieldType fieldType) {
            super(query, fieldType, "l2");
        }

        @Override
        public BiFunction<float[], float[], Float> getScoringMethod(final float[] processedQuery) {
            return (float[] q, float[] v) -> 1 / (1 + KNNScoringUtil.l2Squared(q, v));
        }
    }

    class CosineSimilarity extends LinearSpace {
        public CosineSimilarity(Object query, MappedFieldType fieldType) {
            super(query, fieldType, "cosine");
        }

        @Override
        protected BiFunction<float[], float[], Float> getScoringMethod(final float[] processedQuery) {
            SpaceType.COSINESIMIL.validateVector(processedQuery);
            float qVectorSquaredMagnitude = getVectorMagnitudeSquared(processedQuery);
            return (float[] q, float[] v) -> 1 + KNNScoringUtil.cosinesimilOptimized(q, v, qVectorSquaredMagnitude);
        }
    }

    class L1 extends LinearSpace {
        public L1(Object query, MappedFieldType fieldType) {
            super(query, fieldType, "l1");
        }

        @Override
        protected BiFunction<float[], float[], Float> getScoringMethod(final float[] processedQuery) {
            return (float[] q, float[] v) -> 1 / (1 + KNNScoringUtil.l1Norm(q, v));
        }
    }

    class LInf extends LinearSpace {
        public LInf(Object query, MappedFieldType fieldType) {
            super(query, fieldType, "l-inf");
        }

        @Override
        protected BiFunction<float[], float[], Float> getScoringMethod(final float[] processedQuery) {
            return (float[] q, float[] v) -> 1 / (1 + KNNScoringUtil.lInfNorm(q, v));
        }
    }

    class InnerProd extends LinearSpace {
        public InnerProd(Object query, MappedFieldType fieldType) {
            super(query, fieldType, "innerproduct");
        }

        @Override
        protected BiFunction<float[], float[], Float> getScoringMethod(final float[] processedQuery) {
            return (float[] q, float[] v) -> KNNWeight.normalizeScore(-KNNScoringUtil.innerProduct(q, v));
        }
    }

    class HammingBit implements KNNScoringSpace {

        Object processedQuery;
        BiFunction<?, ?, Float> scoringMethod;

        /**
         * Constructor for HammingBit scoring space. HammingBit scoring space expects values to either be of type
         * Long or Base64 encoded strings, or KNN type with binary data type.
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
            } else if (isKNNVectorFieldType(fieldType)) {
                KNNVectorFieldMapper.KNNVectorFieldType knnVectorFieldType = (KNNVectorFieldMapper.KNNVectorFieldType) fieldType;
                if (isBinaryVectorDataType(knnVectorFieldType) == false) {
                    throw new IllegalArgumentException(
                        String.format(
                            Locale.ROOT,
                            "Incompatible field_type for hamming space. The data type should be binary but got %s",
                            knnVectorFieldType.getVectorDataType().getValue()
                        )
                    );
                }
                this.processedQuery = parseToFloatArray(
                    query,
                    KNNVectorFieldMapperUtil.getExpectedVectorLength(knnVectorFieldType),
                    ((KNNVectorFieldMapper.KNNVectorFieldType) fieldType).getVectorDataType()
                );
                // TODO we want to avoid converting back and forth between byte and float
                this.scoringMethod = (float[] q, float[] v) -> 1 / (1 + KNNScoringUtil.calculateHammingBit(toByte(q), toByte(v)));
            } else {
                throw new IllegalArgumentException(
                    "Incompatible field_type for hamming space. The field type must of type long, binary, or knn_vector."
                );
            }
        }

        private byte[] toByte(final float[] vector) {
            byte[] bytes = new byte[vector.length];
            for (int i = 0; i < vector.length; i++) {
                bytes[i] = (byte) vector[i];
            }
            return bytes;
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
            } else if (this.processedQuery instanceof BigInteger) {
                return new KNNScoreScript.BigIntegerType(
                    params,
                    (BigInteger) this.processedQuery,
                    field,
                    (BiFunction<BigInteger, BigInteger, Float>) this.scoringMethod,
                    lookup,
                    ctx,
                    searcher
                );
            } else {
                return new KNNScoreScript.KNNVectorType(
                    params,
                    (float[]) this.processedQuery,
                    field,
                    (BiFunction<float[], float[], Float>) this.scoringMethod,
                    lookup,
                    ctx,
                    searcher
                );
            }
        }
    }
}
