/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.svs;

import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.Encoder;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.Parameter;
import org.opensearch.knn.index.engine.faiss.AbstractFaissMethod;
import org.opensearch.knn.index.engine.faiss.FaissFlatEncoder;
import org.opensearch.knn.sandbox.ExperimentalAlgorithm;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import static org.opensearch.knn.common.KNNConstants.ENCODER_FLAT;
import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.FAISS_FLAT_DESCRIPTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.sandbox.svs.SVSConstants.DEFAULT_CONSTRUCTION_WINDOW_SIZE;
import static org.opensearch.knn.sandbox.svs.SVSConstants.FAISS_SVS_ENCODER_LVQ;
import static org.opensearch.knn.sandbox.svs.SVSConstants.FAISS_SVS_VAMANA_DESCRIPTION;
import static org.opensearch.knn.sandbox.svs.SVSConstants.METHOD_PARAMETER_ALPHA;
import static org.opensearch.knn.sandbox.svs.SVSConstants.METHOD_PARAMETER_CONSTRUCTION_WINDOW_SIZE;
import static org.opensearch.knn.sandbox.svs.SVSConstants.METHOD_PARAMETER_DEGREE;
import static org.opensearch.knn.sandbox.svs.SVSConstants.METHOD_PARAMETER_SEARCH_BUFFER_CAPACITY;
import static org.opensearch.knn.sandbox.svs.SVSConstants.METHOD_PARAMETER_SEARCH_WINDOW_SIZE;
import static org.opensearch.knn.sandbox.svs.SVSConstants.METHOD_SVS_VAMANA;

/**
 * SVS Vamana method: graph-based approximate search using the Vamana algorithm (Subramanya et al.).
 */
@ExperimentalAlgorithm(description = "Intel SVS Vamana graph-based ANN method", since = "3.7.0")
public class FaissSVSVamanaMethod extends AbstractFaissMethod {

    private static final Set<VectorDataType> SUPPORTED_DATA_TYPES = Set.of(VectorDataType.FLOAT);

    public final static List<SpaceType> SUPPORTED_SPACES = Arrays.asList(SpaceType.L2, SpaceType.INNER_PRODUCT, SpaceType.COSINESIMIL);

    // FLAT, SQ (fp16/sq8), and LVQ. LeanVec is excluded: it requires model training, out of scope for the sandbox.
    public final static Map<String, Encoder> SUPPORTED_ENCODERS = Map.of(
        ENCODER_FLAT,
        new FaissFlatEncoder(),
        ENCODER_SQ,
        new FaissSVSSQEncoder(),
        FAISS_SVS_ENCODER_LVQ,
        new FaissSVSLVQEncoder()
    );

    private final static MethodComponentContext DEFAULT_ENCODER_CONTEXT = new MethodComponentContext(ENCODER_FLAT, Collections.emptyMap());

    public final static MethodComponent METHOD_COMPONENT = initMethodComponent();

    public FaissSVSVamanaMethod() {
        super(METHOD_COMPONENT, Set.copyOf(SUPPORTED_SPACES), new FaissSVSVamanaSearchContext());
    }

    private static MethodComponent initMethodComponent() {
        return MethodComponent.Builder.builder(METHOD_SVS_VAMANA)
            .addSupportedDataTypes(SUPPORTED_DATA_TYPES)
            .addParameter(
                METHOD_PARAMETER_DEGREE,
                new Parameter.IntegerParameter(METHOD_PARAMETER_DEGREE, 64, (v, context) -> v > 0 && v <= 256)
            )
            .addParameter(
                METHOD_PARAMETER_CONSTRUCTION_WINDOW_SIZE,
                new Parameter.IntegerParameter(
                    METHOD_PARAMETER_CONSTRUCTION_WINDOW_SIZE,
                    DEFAULT_CONSTRUCTION_WINDOW_SIZE,
                    (v, context) -> v > 0
                )
            )
            .addParameter(
                METHOD_PARAMETER_SEARCH_WINDOW_SIZE,
                new Parameter.IntegerParameter(METHOD_PARAMETER_SEARCH_WINDOW_SIZE, 10, (v, context) -> v > 0)
            )
            .addParameter(
                METHOD_PARAMETER_SEARCH_BUFFER_CAPACITY,
                new Parameter.IntegerParameter(METHOD_PARAMETER_SEARCH_BUFFER_CAPACITY, 10, (v, context) -> v > 0)
            )
            // alpha controls Vamana graph pruning aggressiveness. Default is null: when unset the SVS
            // runtime applies a metric-dependent default (1.2 for L2, 0.95 for inner product).
            .addParameter(METHOD_PARAMETER_ALPHA, new Parameter.DoubleParameter(METHOD_PARAMETER_ALPHA, null, (v, context) -> v > 0))
            .addParameter(
                METHOD_ENCODER_PARAMETER,
                new Parameter.MethodComponentContextParameter(
                    METHOD_ENCODER_PARAMETER,
                    DEFAULT_ENCODER_CONTEXT,
                    SUPPORTED_ENCODERS.values().stream().collect(Collectors.toMap(Encoder::getName, Encoder::getMethodComponent))
                )
            )
            .setKnnLibraryIndexingContextGenerator(((methodComponent, methodComponentContext, knnMethodConfigContext) -> {
                SvsMethodAsMapBuilder methodAsMapBuilder = SvsMethodAsMapBuilder.builder(
                    FAISS_SVS_VAMANA_DESCRIPTION,
                    methodComponent,
                    methodComponentContext,
                    knnMethodConfigContext
                );
                methodAsMapBuilder.addParameter(METHOD_PARAMETER_DEGREE, "", "");

                // Append the encoder for ALL encoders (incl. default flat) so its MethodComponentContext is
                // normalized into a serializable sub-map; otherwise the raw context fails to serialize into the
                // field mapping. Flat yields a trailing ",Flat" that SVS's native factory rejects, so drop it.
                methodAsMapBuilder.addParameter(METHOD_ENCODER_PARAMETER, ",", "");
                methodAsMapBuilder.dropTrailingDescriptionToken(FAISS_FLAT_DESCRIPTION);

                return methodAsMapBuilder.build();
            }))
            .build();
    }
}
