/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.SegmentReadState;

import java.util.List;

/**
 * Factory for creating {@link PerFieldDerivedVectorInjector} instances.
 */
class PerFieldDerivedVectorInjectorFactory {

    /**
     * Create a {@link PerFieldDerivedVectorInjector} instance based on information in field info.
     *
     * @param fieldInfo FieldInfo for the field to create the injector for
     * @param nestedLineage Nested lineage for the field
     * @param derivedSourceReaders {@link DerivedSourceReaders} instance
     * @return PerFieldDerivedVectorInjector instance
     */
    public static PerFieldDerivedVectorInjector create(
        FieldInfo fieldInfo,
        List<String> nestedLineage,
        DerivedSourceReaders derivedSourceReaders,
        SegmentReadState segmentReadState
    ) {
        // Non-nested case - the doc has no nested lineage, so it is not nested
        if (nestedLineage == null || nestedLineage.isEmpty()) {
            return new RootPerFieldDerivedVectorInjector(fieldInfo, derivedSourceReaders);
        }

        // Nested case
        return new NestedPerFieldDerivedVectorInjector(fieldInfo, nestedLineage, derivedSourceReaders, segmentReadState);
    }
}
