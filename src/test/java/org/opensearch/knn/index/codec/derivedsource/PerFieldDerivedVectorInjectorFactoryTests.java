/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.KNNCodecTestUtil;

public class PerFieldDerivedVectorInjectorFactoryTests extends KNNTestCase {
    public void testCreate() {
        // Non-nested case
        PerFieldDerivedVectorInjector perFieldDerivedVectorInjector = PerFieldDerivedVectorInjectorFactory.create(
            KNNCodecTestUtil.FieldInfoBuilder.builder("test").build(),
            new DerivedSourceReaders(null, null, null, null, false),
            null
        );
        assertTrue(perFieldDerivedVectorInjector instanceof RootPerFieldDerivedVectorInjector);

        // Nested case
        perFieldDerivedVectorInjector = PerFieldDerivedVectorInjectorFactory.create(
            KNNCodecTestUtil.FieldInfoBuilder.builder("parent.test").build(),
            new DerivedSourceReaders(null, null, null, null, false),
            null
        );
        assertTrue(perFieldDerivedVectorInjector instanceof NestedPerFieldDerivedVectorInjector);
    }
}
