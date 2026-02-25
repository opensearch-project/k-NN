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

package org.opensearch.knn.indices;

import lombok.experimental.UtilityClass;
import org.apache.commons.lang3.StringUtils;

import java.util.Locale;

/**
 * A utility class for models.
 */
@UtilityClass
public class ModelUtil {

    public static void blockCommasInModelDescription(String description) {
        if (description.contains(",")) {
            throw new IllegalArgumentException("Model description cannot contain any commas: ','");
        }
    }

    public static boolean isModelPresent(ModelMetadata modelMetadata) {
        return modelMetadata != null;
    }

    public static boolean isModelCreated(ModelMetadata modelMetadata) {
        if (!isModelPresent(modelMetadata)) {
            return false;
        }
        return modelMetadata.getState().equals(ModelState.CREATED);
    }

    /**
     * Gets Model Metadata from a given model id.
     * @param modelId {@link String}
     * @return {@link ModelMetadata}
     */
    public static ModelMetadata getModelMetadata(final String modelId) {
        if (StringUtils.isEmpty(modelId)) {
            return null;
        }
        ModelDao modelDao = ModelDao.OpenSearchKNNModelDao.getInstance();
        final ModelMetadata modelMetadata = modelDao.getMetadata(modelId);
        if (isModelCreated(modelMetadata) == false) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "Model ID '%s' is not created.", modelId));
        }
        return modelMetadata;
    }

}
