/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.script;

import lombok.Getter;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.IndexSearcher;
import org.opensearch.Version;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapperUtil;
import org.opensearch.knn.index.mapper.KNNVectorFieldType;
import org.opensearch.knn.index.query.KNNWeight;
import org.opensearch.script.ScoreScript;
import org.opensearch.search.lookup.SearchLookup;

import java.io.IOException;
import java.math.BigInteger;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.function.BiFunction;

import static org.opensearch.knn.plugin.script.KNNScoringSpaceUtil.getVectorMagnitudeSquared;
import static org.opensearch.knn.plugin.script.KNNScoringSpaceUtil.isBinaryFieldType;
import static org.opensearch.knn.plugin.script.KNNScoringSpaceUtil.isKNNVectorFieldType;
import static org.opensearch.knn.plugin.script.KNNScoringSpaceUtil.isLongFieldType;
import static org.opensearch.knn.plugin.script.KNNScoringSpaceUtil.parseToBigInteger;
import static org.opensearch.knn.plugin.script.KNNScoringSpaceUtil.parseToFloatArray;
import static org.opensearch.knn.plugin.script.KNNScoringSpaceUtil.parseToByteArray;
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
     * Base class to represent vector space for knn field
     */
    abstract class KNNFieldSpace implements KNNScoringSpace {
        public static final Set<VectorDataType> DATA_TYPES_DEFAULT = Set.of(VectorDataType.FLOAT, VectorDataType.BYTE);

        private Object processedQuery;
        @Getter
        private BiFunction<?, ?, Float> scoringMethod;

        public KNNFieldSpace(final Object query, final MappedFieldType fieldType, final String spaceName) {
            this(query, fieldType, spaceName, DATA_TYPES_DEFAULT);
        }

        public KNNFieldSpace(
            final Object query,
            final MappedFieldType fieldType,
            final String spaceName,
            final Set<VectorDataType> supportingVectorDataTypes
        ) {
            KNNVectorFieldType knnVectorFieldType = toKNNVectorFieldType(fieldType, spaceName, supportingVectorDataTypes);
            this.processedQuery = getProcessedQuery(query, knnVectorFieldType);
            this.scoringMethod = getScoringMethod(this.processedQuery, knnVectorFieldType.getKnnMappingConfig().getIndexCreatedVersion());
        }

        @SuppressWarnings("unchecked")
        public ScoreScript getScoreScript(
            Map<String, Object> params,
            String field,
            SearchLookup lookup,
            LeafReaderContext ctx,
            IndexSearcher searcher
        ) throws IOException {
            if (processedQuery instanceof float[]) {
                return new KNNScoreScript.KNNFloatVectorType(
                    params,
                    (float[]) this.processedQuery,
                    field,
                    (BiFunction<float[], float[], Float>) this.scoringMethod,
                    lookup,
                    ctx,
                    searcher
                );
            } else if (processedQuery instanceof byte[]) {
                return new KNNScoreScript.KNNByteVectorType(
                    params,
                    (byte[]) this.processedQuery,
                    field,
                    (BiFunction<byte[], byte[], Float>) this.scoringMethod,
                    lookup,
                    ctx,
                    searcher
                );
            } else {
                throw new IllegalStateException(
                    "Unexpected type for processedQuery. Expected float[] or byte[], but got: " + processedQuery.getClass().getName()
                );
            }
        }

        private KNNVectorFieldType toKNNVectorFieldType(
            final MappedFieldType fieldType,
            final String spaceName,
            final Set<VectorDataType> supportingVectorDataTypes
        ) {
            if (isKNNVectorFieldType(fieldType) == false) {
                throw new IllegalArgumentException(
                    String.format(Locale.ROOT, "Incompatible field_type for %s space. The field type must be knn_vector.", spaceName)
                );
            }

            KNNVectorFieldType knnVectorFieldType = (KNNVectorFieldType) fieldType;
            VectorDataType vectorDataType = knnVectorFieldType.getVectorDataType() == null
                ? VectorDataType.FLOAT
                : knnVectorFieldType.getVectorDataType();
            if (supportingVectorDataTypes.contains(vectorDataType) == false) {
                throw new IllegalArgumentException(
                    String.format(
                        Locale.ROOT,
                        "Incompatible field_type for %s space. The data type should be %s but got %s",
                        spaceName,
                        supportingVectorDataTypes,
                        vectorDataType
                    )
                );
            }

            return knnVectorFieldType;
        }

        protected Object getProcessedQuery(final Object query, final KNNVectorFieldType knnVectorFieldType) {
            VectorDataType vectorDataType = knnVectorFieldType.getVectorDataType() == null
                ? VectorDataType.FLOAT
                : knnVectorFieldType.getVectorDataType();
            if (vectorDataType == VectorDataType.FLOAT) {
                return parseToFloatArray(
                    query,
                    KNNVectorFieldMapperUtil.getExpectedVectorLength(knnVectorFieldType),
                    knnVectorFieldType.getVectorDataType()
                );
            }
            return parseToByteArray(
                query,
                KNNVectorFieldMapperUtil.getExpectedVectorLength(knnVectorFieldType),
                knnVectorFieldType.getVectorDataType()
            );
        }

        public abstract BiFunction<?, ?, Float> getScoringMethod(final Object processedQuery);

        protected BiFunction<?, ?, Float> getScoringMethod(final Object processedQuery, Version indexCreatedVersion) {
            return getScoringMethod(processedQuery);
        }

    }

    class L2 extends KNNFieldSpace {
        public L2(final Object query, final MappedFieldType fieldType) {
            super(query, fieldType, "l2");
        }

        @Override
        public BiFunction<?, ?, Float> getScoringMethod(final Object processedQuery) {
            if (processedQuery instanceof float[]) {
                return (float[] q, float[] v) -> 1 / (1 + KNNScoringUtil.l2Squared(q, v));
            } else {
                return (byte[] q, byte[] v) -> 1 / (1 + KNNScoringUtil.l2Squared(q, v));
            }
        }
    }

    class CosineSimilarity extends KNNFieldSpace {
        public CosineSimilarity(Object query, MappedFieldType fieldType) {
            super(query, fieldType, "cosine");
        }

        @Override
        public BiFunction<?, ?, Float> getScoringMethod(Object processedQuery) {
            return getScoringMethod(processedQuery, Version.CURRENT);
        }

        @Override
        protected BiFunction<?, ?, Float> getScoringMethod(final Object processedQuery, Version indexCreatedVersion) {
            if (processedQuery instanceof float[]) {
                SpaceType.COSINESIMIL.validateVector((float[]) processedQuery);
                float qVectorSquaredMagnitude = getVectorMagnitudeSquared((float[]) processedQuery);
                if (indexCreatedVersion.onOrAfter(Version.V_2_19_0)) {
                    // To be consistent, we will be using same formula used by lucene as mentioned below
                    // https://github.com/apache/lucene/blob/0494c824e0ac8049b757582f60d085932a890800/lucene/core/src/java/org/apache/lucene/index/VectorSimilarityFunction.java#L73
                    // for indices that are created on or after 2.19.0
                    //
                    // OS Score = ( 2 − cosineSimil) / 2
                    // However cosineSimil = 1 - cos θ, after applying this to above formula,
                    // OS Score = ( 2 − ( 1 − cos θ ) ) / 2
                    // which simplifies to
                    // OS Score = ( 1 + cos θ ) / 2
                    return (float[] q, float[] v) -> Math.max(
                        ((1 + KNNScoringUtil.cosinesimilOptimized(q, v, qVectorSquaredMagnitude)) / 2.0F),
                        0
                    );
                }
                return (float[] q, float[] v) -> 1 + KNNScoringUtil.cosinesimilOptimized(q, v, qVectorSquaredMagnitude);
            } else {
                SpaceType.COSINESIMIL.validateVector((byte[]) processedQuery);
                return (byte[] q, byte[] v) -> 1 + KNNScoringUtil.cosinesimil(q, v);
            }
        }
    }

    class L1 extends KNNFieldSpace {
        public L1(Object query, MappedFieldType fieldType) {
            super(query, fieldType, "l1");
        }

        @Override
        public BiFunction<?, ?, Float> getScoringMethod(final Object processedQuery) {
            if (processedQuery instanceof float[]) {
                return (float[] q, float[] v) -> 1 / (1 + KNNScoringUtil.l1Norm(q, v));
            } else {
                return (byte[] q, byte[] v) -> 1 / (1 + KNNScoringUtil.l1Norm(q, v));
            }
        }
    }

    class LInf extends KNNFieldSpace {
        public LInf(Object query, MappedFieldType fieldType) {
            super(query, fieldType, "l-inf");
        }

        @Override
        public BiFunction<?, ?, Float> getScoringMethod(final Object processedQuery) {
            if (processedQuery instanceof float[]) {
                return (float[] q, float[] v) -> 1 / (1 + KNNScoringUtil.lInfNorm(q, v));
            } else {
                return (byte[] q, byte[] v) -> 1 / (1 + KNNScoringUtil.lInfNorm(q, v));
            }
        }
    }

    class InnerProd extends KNNFieldSpace {
        public InnerProd(Object query, MappedFieldType fieldType) {
            super(query, fieldType, "innerproduct");
        }

        @Override
        public BiFunction<?, ?, Float> getScoringMethod(final Object processedQuery) {
            if (processedQuery instanceof float[]) {
                return (float[] q, float[] v) -> KNNWeight.normalizeScore(-KNNScoringUtil.innerProduct(q, v));
            } else {
                return (byte[] q, byte[] v) -> KNNWeight.normalizeScore(-KNNScoringUtil.innerProduct(q, v));
            }
        }
    }

    class Hamming extends KNNFieldSpace {
        private static final Set<VectorDataType> DATA_TYPES_HAMMING = Set.of(VectorDataType.BINARY);

        public Hamming(Object query, MappedFieldType fieldType) {
            super(query, fieldType, "hamming", DATA_TYPES_HAMMING);
        }

        @Override
        public BiFunction<?, ?, Float> getScoringMethod(final Object processedQuery) {
            return (byte[] q, byte[] v) -> 1 / (1 + KNNScoringUtil.calculateHammingBit(q, v));
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
                    "Incompatible field_type for hammingbit space. The field type must of type long or binary."
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
}
