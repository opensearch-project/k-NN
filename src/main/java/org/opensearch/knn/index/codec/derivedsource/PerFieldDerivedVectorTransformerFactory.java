/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import org.apache.lucene.index.FieldInfo;

public class PerFieldDerivedVectorTransformerFactory {

    /**
     * Create a {@link PerFieldDerivedVectorTransformer} instance based on information in field info.
     *
     * @param fieldInfo FieldInfo for the field to create the injector for
     * @param derivedSourceReaders {@link DerivedSourceReaders} instance
     * @return PerFieldDerivedVectorInjector instance
     */
    public static PerFieldDerivedVectorTransformer create(
        FieldInfo fieldInfo,
        boolean isNested,
        DerivedSourceReaders derivedSourceReaders
    ) {
        // Nested case
        if (isNested) {
            return new NestedPerFieldDerivedVectorTransformer(fieldInfo, derivedSourceReaders);
        }

        // Non-nested case
        return new RootPerFieldDerivedVectorTransformer(fieldInfo, derivedSourceReaders);
    }

}
