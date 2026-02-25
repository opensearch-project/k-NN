/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.backward_codecs.KNN9120Codec;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.KNNCodecTestUtil;

public class PerFieldDerivedVectorInjectorFactoryTests extends KNNTestCase {
    public void testCreate() {
        // Non-nested case
        PerFieldDerivedVectorInjector perFieldDerivedVectorInjector = PerFieldDerivedVectorInjectorFactory.create(
            KNNCodecTestUtil.FieldInfoBuilder.builder("test").build(),
            new KNN9120DerivedSourceReaders(null, null, null, null),
            null
        );
        assertTrue(perFieldDerivedVectorInjector instanceof RootPerFieldDerivedVectorInjector);

        // Nested case
        perFieldDerivedVectorInjector = PerFieldDerivedVectorInjectorFactory.create(
            KNNCodecTestUtil.FieldInfoBuilder.builder("parent.test").build(),
            new KNN9120DerivedSourceReaders(null, null, null, null),
            null
        );
        assertTrue(perFieldDerivedVectorInjector instanceof NestedPerFieldDerivedVectorInjector);
    }
}
