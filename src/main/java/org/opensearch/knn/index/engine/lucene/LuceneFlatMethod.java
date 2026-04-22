/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.lucene;

import com.google.common.collect.ImmutableSet;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.AbstractKNNMethod;
import org.opensearch.knn.index.engine.MethodComponent;

import java.util.Arrays;
import java.util.List;
import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.METHOD_FLAT;

/**
 * Lucene Flat implementation
 */
public class LuceneFlatMethod extends AbstractKNNMethod {

    private static final Set<VectorDataType> SUPPORTED_DATA_TYPES = ImmutableSet.of(VectorDataType.FLOAT);

    public final static List<SpaceType> SUPPORTED_SPACES = Arrays.asList(
        SpaceType.UNDEFINED,
        SpaceType.L2,
        SpaceType.COSINESIMIL,
        SpaceType.INNER_PRODUCT
    );

    final static MethodComponent FLAT_METHOD_COMPONENT = initMethodComponent();

    /**
     * Constructor for LuceneFlatMethod
     *
     * @see AbstractKNNMethod
     */
    public LuceneFlatMethod() {
        super(FLAT_METHOD_COMPONENT, Set.copyOf(SUPPORTED_SPACES), new LuceneFlatSearchContext());
    }

    private static MethodComponent initMethodComponent() {
        return MethodComponent.Builder.builder(METHOD_FLAT).addSupportedDataTypes(SUPPORTED_DATA_TYPES).build();
    }

}
