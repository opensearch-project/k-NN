/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.lucene;

import com.google.common.collect.ImmutableMap;
import org.apache.lucene.util.Version;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.JVMLibrary;
import org.opensearch.knn.index.engine.KNNMethod;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodResolver;
import org.opensearch.knn.index.engine.ResolvedMethodContext;

import java.util.List;
import java.util.Map;
import java.util.function.Function;

import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;

/**
 * KNN Library for Lucene
 */
public class Lucene extends JVMLibrary {

    Map<SpaceType, Function<Float, Float>> distanceTransform;

    final static Map<String, KNNMethod> METHODS = ImmutableMap.of(METHOD_HNSW, new LuceneHNSWMethod());

    // Map that overrides the default distance translations for Lucene, check more details in knn documentation:
    // https://opensearch.org/docs/latest/search-plugins/knn/approximate-knn/#spaces
    private final static Map<SpaceType, Function<Float, Float>> DISTANCE_TRANSLATIONS = ImmutableMap.<
        SpaceType,
        Function<Float, Float>>builder()
        .put(SpaceType.COSINESIMIL, distance -> (2 - distance) / 2)
        .put(SpaceType.INNER_PRODUCT, distance -> distance <= 0 ? 1 / (1 - distance) : distance + 1)
        .build();

    public final static Lucene INSTANCE = new Lucene(METHODS, Version.LATEST.toString(), DISTANCE_TRANSLATIONS);

    private final MethodResolver methodResolver;

    /**
     * Constructor
     *
     * @param methods Map of k-NN methods that the library supports
     * @param version String representing version of library
     * @param distanceTransform Map of space type to distance transformation function
     */
    Lucene(Map<String, KNNMethod> methods, String version, Map<SpaceType, Function<Float, Float>> distanceTransform) {
        super(methods, version);
        this.distanceTransform = distanceTransform;
        this.methodResolver = new LuceneMethodResolver();
    }

    @Override
    public String getExtension() {
        throw new UnsupportedOperationException("Getting extension for Lucene is not supported");
    }

    @Override
    public String getCompoundExtension() {
        throw new UnsupportedOperationException("Getting compound extension for Lucene is not supported");
    }

    @Override
    public float score(float rawScore, SpaceType spaceType) {
        // The score returned by Lucene follows the higher the score, the better the result convention. It will
        // actually invert the distance score so that a higher number is a better score. So, we can just return the
        // score provided.
        return rawScore;
    }

    @Override
    public Float distanceToRadialThreshold(Float distance, SpaceType spaceType) {
        // Lucene requires score threshold to be parameterized when calling the radius search.
        if (this.distanceTransform.containsKey(spaceType)) {
            return this.distanceTransform.get(spaceType).apply(distance);
        }
        return spaceType.scoreTranslation(distance);
    }

    @Override
    public Float scoreToRadialThreshold(Float score, SpaceType spaceType) {
        // Lucene engine uses distance as is and does not need transformation
        return score;
    }

    @Override
    public List<String> mmapFileExtensions() {
        return List.of("vec", "vex");
    }

    @Override
    public ResolvedMethodContext resolveMethod(
        KNNMethodContext knnMethodContext,
        KNNMethodConfigContext knnMethodConfigContext,
        boolean shouldRequireTraining,
        final SpaceType spaceType
    ) {
        return methodResolver.resolveMethod(knnMethodContext, knnMethodConfigContext, shouldRequireTraining, spaceType);
    }
}
