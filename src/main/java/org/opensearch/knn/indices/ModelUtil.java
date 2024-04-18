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

/**
 * A utility class for models.
 */
public class ModelUtil {

    public static boolean isModelPresent(ModelMetadata modelMetadata) {
        return modelMetadata != null;
    }

    public static boolean isModelCreated(ModelMetadata modelMetadata) {
        if (!isModelPresent(modelMetadata)) {
            return false;
        }
        return modelMetadata.getState().equals(ModelState.CREATED);
    }

}
