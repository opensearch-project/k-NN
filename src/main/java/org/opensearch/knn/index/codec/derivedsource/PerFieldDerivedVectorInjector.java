/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import java.io.IOException;
import java.util.Map;

/**
 * Interface for injecting derived vectors into a source map per field.
 */
public interface PerFieldDerivedVectorInjector {

    /**
     * Injects the derived vector for this field into the sourceAsMap. Implementing classes must handle the case where
     * a document does not have a value for their field.
     *
     * @param docId Document ID
     * @param sourceAsMap Source as map
     * @throws IOException if there is an issue reading from the formats
     */
    void inject(int docId, Map<String, Object> sourceAsMap) throws IOException;
}
