/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import java.io.IOException;
import java.util.function.Function;

public interface PerFieldDerivedVectorTransformer extends Function<Object, Object> {

    /**
     * Update the current doc to the given doc id
     *
     * @param offset Offset to advance iterators to
     * @param docId Parent doc id
     * @throws IOException thrown on invalid read
     */
    void setCurrentDoc(int offset, int docId) throws IOException;
}
