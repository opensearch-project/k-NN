/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.index.mapper;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.util.BytesRef;
import org.opensearch.Version;
import org.opensearch.common.settings.Settings;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.KnnCircuitBreakerException;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.util.KNNVectorSerializerFactory;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.util.IndexHyperParametersUtil;

import java.util.Arrays;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.HNSW_ALGO_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.HNSW_ALGO_M;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_M;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;

/**
 * Utility class for KNNVectorFieldMapper
 */
@Log4j2
@NoArgsConstructor(access = AccessLevel.PRIVATE)
public class KNNVectorFieldMapperUtil {

    /**
     * @param knnEngine  KNNEngine
     * @return  DocValues FieldType of type Binary
     */
    public static FieldType buildDocValuesFieldType(KNNEngine knnEngine) {
        FieldType field = new FieldType();
        field.putAttribute(KNN_ENGINE, knnEngine.getName());
        field.setDocValuesType(DocValuesType.BINARY);
        field.freeze();
        return field;
    }

    /**
     * Creates a stored field for a byte vector
     *
     * @param name field name
     * @param vector vector to be added to stored field
     */
    public static StoredField createStoredFieldForByteVector(String name, byte[] vector) {
        return new StoredField(name, vector);
    }

    /**
     * Creates a stored field for a float vector
     *
     * @param name field name
     * @param vector vector to be added to stored field
     */
    public static StoredField createStoredFieldForFloatVector(String name, float[] vector) {
        return new StoredField(name, KNNVectorSerializerFactory.getDefaultSerializer().floatToByteArray(vector));
    }

    /**
     * @param storedVector Vector representation in bytes
     * @param vectorDataType type of vector
     * @return either int[] or float[] of corresponding vector
     */
    public static Object deserializeStoredVector(BytesRef storedVector, VectorDataType vectorDataType) {
        if (VectorDataType.BYTE == vectorDataType) {
            byte[] bytes = storedVector.bytes;
            int[] byteAsIntArray = new int[bytes.length];
            Arrays.setAll(byteAsIntArray, i -> bytes[i]);
            return byteAsIntArray;
        }

        return vectorDataType.getVectorFromBytesRef(storedVector);
    }

    /**
     * Get the expected vector length from a specified knn vector field type.
     *
     * If the field is model-based, get dimensions from model metadata.
     * For binary vector, the expected vector length is dimension divided by 8
     *
     * @param knnVectorFieldType knn vector field type
     * @return expected vector length
     */
    public static int getExpectedVectorLength(final KNNVectorFieldType knnVectorFieldType) {
        int expectedDimensions = knnVectorFieldType.getKnnMappingConfig().getDimension();
        return VectorDataType.BINARY == knnVectorFieldType.getVectorDataType() ? expectedDimensions / 8 : expectedDimensions;
    }

    /**
     * Validate if the circuit breaker is triggered
     */
    static void validateIfCircuitBreakerIsNotTriggered() {
        if (KNNSettings.isCircuitBreakerTriggered()) {
            throw new KnnCircuitBreakerException(
                "Parsing the created knn vector fields prior to indexing has failed as the circuit breaker triggered.  This indicates that the cluster is low on memory resources and cannot index more documents at the moment. Check _plugins/_knn/stats for the circuit breaker status."
            );
        }
    }

    /**
     * Validate if plugin is enabled
     */
    static void validateIfKNNPluginEnabled() {
        if (!KNNSettings.isKNNPluginEnabled()) {
            throw new IllegalStateException("KNN plugin is disabled. To enable update knn.plugin.enabled setting to true");
        }
    }

    /**
     * Prerequisite: Index should a knn index which is validated via index settings index.knn setting. This function
     * assumes that caller has already validated that index is a KNN index.
     * We will use LuceneKNNVectorsFormat when these below condition satisfy:
     * <ol>
     *  <li>Index is created with Version of opensearch >= 2.17</li>
     *  <li>Cluster setting is enabled to use Lucene KNNVectors format. This condition is temporary condition and will be
     * removed before release.</li>
     * </ol>
     * @param indexCreatedVersion {@link Version}
     * @return true if vector field should use KNNVectorsFormat
     */
    static boolean useLuceneKNNVectorsFormat(final Version indexCreatedVersion) {
        return indexCreatedVersion.onOrAfter(Version.V_2_17_0);
    }

    public static SpaceType getSpaceType(final Settings indexSettings) {
        String spaceType = indexSettings.get(KNNSettings.INDEX_KNN_SPACE_TYPE.getKey());
        if (spaceType == null) {
            spaceType = KNNSettings.INDEX_KNN_DEFAULT_SPACE_TYPE;
            log.info(
                String.format(
                    "[KNN] The setting \"%s\" was not set for the index. Likely caused by recent version upgrade. Setting the setting to the default value=%s",
                    METHOD_PARAMETER_SPACE_TYPE,
                    spaceType
                )
            );
        }
        return SpaceType.getSpace(spaceType);
    }

    private static int getM(Settings indexSettings) {
        String m = indexSettings.get(KNNSettings.INDEX_KNN_ALGO_PARAM_M_SETTING.getKey());
        if (m == null) {
            log.info(
                String.format(
                    "[KNN] The setting \"%s\" was not set for the index. Likely caused by recent version upgrade. Setting the setting to the default value=%s",
                    HNSW_ALGO_M,
                    KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_M
                )
            );
            return KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_M;
        }
        return Integer.parseInt(m);
    }

    private static int getEfConstruction(Settings indexSettings, Version indexVersion) {
        final String efConstruction = indexSettings.get(KNNSettings.INDEX_KNN_ALGO_PARAM_EF_CONSTRUCTION_SETTING.getKey());
        if (efConstruction == null) {
            final int defaultEFConstructionValue = IndexHyperParametersUtil.getHNSWEFConstructionValue(indexVersion);
            log.info(
                String.format(
                    "[KNN] The setting \"%s\" was not set for the index. Likely caused by recent version upgrade. "
                        + "Picking up default value for the index =%s",
                    HNSW_ALGO_EF_CONSTRUCTION,
                    defaultEFConstructionValue
                )
            );
            return defaultEFConstructionValue;
        }
        return Integer.parseInt(efConstruction);
    }

    static KNNMethodContext createKNNMethodContextFromLegacy(
        Settings indexSettings,
        Version indexCreatedVersion,
        SpaceType resolvedSpaceType
    ) {
        return new KNNMethodContext(
            KNNEngine.NMSLIB,
            resolvedSpaceType,
            new MethodComponentContext(
                METHOD_HNSW,
                Map.of(
                    METHOD_PARAMETER_M,
                    KNNVectorFieldMapperUtil.getM(indexSettings),
                    METHOD_PARAMETER_EF_CONSTRUCTION,
                    KNNVectorFieldMapperUtil.getEfConstruction(indexSettings, indexCreatedVersion)
                )
            )
        );
    }
}
