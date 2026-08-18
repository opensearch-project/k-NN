/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.common.exception;

import org.opensearch.core.rest.RestStatus;
import org.opensearch.knn.KNNTestCase;

public class DeleteModelExceptionTests extends KNNTestCase {

    public void testConstructorWithMessage() {
        String modelId = "test-model-id";
        DeleteModelException exception = new DeleteModelException("Cannot delete model [{}]", modelId);

        assertEquals("Cannot delete model [test-model-id]", exception.getMessage());
    }

    public void testStatusReturnsConflict() {
        DeleteModelException exception = new DeleteModelException("Model is in use");

        assertEquals(RestStatus.CONFLICT, exception.status());
    }

    public void testConstructorWithMultipleArgs() {
        DeleteModelException exception = new DeleteModelException("Model [{}] is in state [{}]", "model-1", "training");

        assertEquals("Model [model-1] is in state [training]", exception.getMessage());
    }
}
