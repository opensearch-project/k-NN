/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.jvector;

import com.google.common.collect.ImmutableSet;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.*;

import java.util.Arrays;
import java.util.List;
import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.*;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;

public class JVectorDiskANNMethod extends AbstractKNNMethod {

    private static final Set<VectorDataType> SUPPORTED_DATA_TYPES = ImmutableSet.of(VectorDataType.FLOAT);

    public final static List<SpaceType> SUPPORTED_SPACES = Arrays.asList(
        SpaceType.UNDEFINED,
        SpaceType.L2,
        SpaceType.L1,
        SpaceType.LINF,
        SpaceType.COSINESIMIL,
        SpaceType.INNER_PRODUCT
    );

    final static MethodComponent HNSW_METHOD_COMPONENT = initMethodComponent();

    private static MethodComponent initMethodComponent() {
        return MethodComponent.Builder.builder(METHOD_HNSW)
            .addSupportedDataTypes(SUPPORTED_DATA_TYPES)
            .addParameter(
                METHOD_PARAMETER_M,
                new Parameter.IntegerParameter(METHOD_PARAMETER_M, KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_M, (v, context) -> v > 0)
            )
            .addParameter(
                METHOD_PARAMETER_EF_CONSTRUCTION,
                new Parameter.IntegerParameter(
                    METHOD_PARAMETER_EF_CONSTRUCTION,
                    KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_EF_CONSTRUCTION,
                    (v, context) -> v > 0
                )
            )
            .build();
    }

    public JVectorDiskANNMethod() {
        super(HNSW_METHOD_COMPONENT, Set.copyOf(SUPPORTED_SPACES), new DefaultHnswSearchContext());
    }
}
