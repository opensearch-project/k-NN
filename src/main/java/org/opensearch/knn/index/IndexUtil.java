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

package org.opensearch.knn.index;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.cluster.metadata.MappingMetadata;
import org.opensearch.common.ValidationException;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.knn.indices.ModelMetadata;

import java.io.File;
import java.util.Collections;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.BYTES_PER_KILOBYTES;
import static org.opensearch.knn.common.KNNConstants.HNSW_ALGO_EF_SEARCH;
import static org.opensearch.knn.common.KNNConstants.SPACE_TYPE;

public class IndexUtil {

    /**
     * Determines the size of a file on disk in kilobytes
     *
     * @param filePath path to the file
     * @return file size in kilobytes
     */
    public static int getFileSizeInKB(String filePath) {
        if (filePath == null || filePath.isEmpty()) {
            return 0;
        }
        File file = new File(filePath);
        if (!file.exists() || !file.isFile()) {
            return 0;
        }

        return Math.toIntExact((file.length() / BYTES_PER_KILOBYTES) + 1L); // Add one so that integer division rounds up
    }

    /**
     * Validate that a field is a k-NN vector field and has the expected dimension
     *
     * @param indexMetadata metadata for index to validate
     * @param field field name to validate
     * @param expectedDimension expected dimension of the field. If this value is negative, dimension will not be
     *                          checked
     * @param modelDao used to look up dimension if field uses a model for initialization. Can be null if
     *                 expectedDimension is negative
     * @return ValidationException exception produced by field validation
     */
    @SuppressWarnings("unchecked")
    public static ValidationException validateKnnField(
        IndexMetadata indexMetadata,
        String field,
        int expectedDimension,
        ModelDao modelDao
    ) {
        // Index metadata should not be null
        if (indexMetadata == null) {
            throw new IllegalArgumentException("IndexMetadata should not be null");
        }

        ValidationException exception = new ValidationException();

        // Check the mapping
        MappingMetadata mappingMetadata = indexMetadata.mapping();
        if (mappingMetadata == null) {
            exception.addValidationError("Invalid index. Index does not contain a mapping");
            return exception;
        }

        // The mapping output *should* look like this:
        // "{properties={field={type=knn_vector, dimension=8}}}"
        Map<String, Object> properties = (Map<String, Object>) mappingMetadata.getSourceAsMap().get("properties");

        if (properties == null) {
            exception.addValidationError("Properties in map does not exists. This is unexpected");
            return exception;
        }

        Object fieldMapping = properties.get(field);

        // Check field existence
        if (fieldMapping == null) {
            exception.addValidationError(String.format("Field \"%s\" does not exist.", field));
            return exception;
        }

        // Check if field is a map. If not, that is a problem
        if (!(fieldMapping instanceof Map)) {
            exception.addValidationError(String.format("Field info for \"%s\" is not a map.", field));
            return exception;
        }

        Map<String, Object> fieldMap = (Map<String, Object>) fieldMapping;

        // Check fields type is knn_vector
        Object type = fieldMap.get("type");

        if (!(type instanceof String) || !KNNVectorFieldMapper.CONTENT_TYPE.equals(type)) {
            exception.addValidationError(String.format("Field \"%s\" is not of type %s.", field, KNNVectorFieldMapper.CONTENT_TYPE));
            return exception;
        }

        // Return if dimension does not need to be checked
        if (expectedDimension < 0) {
            return null;
        }

        // Check that the dimension of the method passed in matches that of the model
        Object dimension = fieldMap.get(KNNConstants.DIMENSION);

        // If dimension is null, the training index/field could use a model. In this case, we need to get the model id
        // for the index and then fetch its dimension from the models metadata
        if (dimension == null) {

            String modelId = (String) fieldMap.get(KNNConstants.MODEL_ID);

            if (modelId == null) {
                exception.addValidationError(String.format("Field \"%s\" does not have a dimension set.", field));
                return exception;
            }

            if (modelDao == null) {
                throw new IllegalArgumentException(String.format("Field \"%s\" uses model. modelDao cannot be null.", field));
            }

            ModelMetadata modelMetadata = modelDao.getMetadata(modelId);
            if (modelMetadata == null) {
                exception.addValidationError(String.format("Model \"%s\" for field \"%s\" does not exist.", modelId, field));
                return exception;
            }

            dimension = modelMetadata.getDimension();
            if ((Integer) dimension != expectedDimension) {
                exception.addValidationError(
                    String.format(
                        "Field \"%s\" has dimension %d, which is different from " + "dimension specified in the training request: %d",
                        field,
                        dimension,
                        expectedDimension
                    )
                );
                return exception;
            }

            return null;
        }

        // If the dimension was found in training fields mapping, check that it equals the models proposed dimension.
        if ((Integer) dimension != expectedDimension) {
            exception.addValidationError(
                String.format(
                    "Field \"%s\" has dimension %d, which is different from " + "dimension specified in the training request: %d",
                    field,
                    dimension,
                    expectedDimension
                )
            );
            return exception;
        }

        return null;
    }

    /**
     * Gets the load time parameters for a given engine.
     *
     * @param spaceType Space for this particular segment
     * @param knnEngine Engine used for the native library indices being loaded in
     * @param indexName Name of OpenSearch index that the segment files belong to
     * @return load parameters that will be passed to the JNI.
     */
    public static Map<String, Object> getParametersAtLoading(SpaceType spaceType, KNNEngine knnEngine, String indexName) {
        Map<String, Object> loadParameters = Maps.newHashMap(ImmutableMap.of(SPACE_TYPE, spaceType.getValue()));

        // For nmslib, we need to add the dynamic ef_search parameter that needs to be passed in when the
        // hnsw graphs are loaded into memory
        if (KNNEngine.NMSLIB.equals(knnEngine)) {
            loadParameters.put(HNSW_ALGO_EF_SEARCH, KNNSettings.getEfSearchParam(indexName));
        }

        return Collections.unmodifiableMap(loadParameters);
    }
}
