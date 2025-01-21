/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.jvector;

import com.google.common.collect.ImmutableMap;
import org.apache.lucene.util.Version;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.codec.jvector.JVectorFormat;
import org.opensearch.knn.index.engine.*;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import static org.opensearch.knn.common.KNNConstants.DISK_ANN;

/**
 * JVector engine is a JVM-based k-NN library that leverages the <a href="https://github.com/jbellis/jvector">JVector library</a> .
 * This class provides specific functionalities for handling k-NN methods along with transformations
 * for distance and scoring based on different space types.
 * JVector unique advantage over other libraries are it's very lightweight, SIMD support and implementation of Disk ANN.
 *
 * JVector also manages method resolution during query execution and facilitates interaction with
 * Lucene's k-NN framework by providing conversions between scores and distance thresholds.
 *
 * The class makes use of pre-defined static mappings for supported k-NN methods and distance translations,
 * which determine how scores and distances should be interpreted and converted in various space types.
 */
public class JVector extends JVMLibrary {
    private final static String CUSTOM_COMPOUND_FILE_EXTENSION = "cjvec";
    private final Map<SpaceType, Function<Float, Float>> distanceTransform;
    private final MethodResolver methodResolver = new JVectorMethodResolver();

    public JVector(Map<String, KNNMethod> methods, String version, Map<SpaceType, Function<Float, Float>> distanceTransform) {
        super(methods, version);
        this.distanceTransform = distanceTransform;
    }

    final static Map<String, KNNMethod> METHODS = Map.of(DISK_ANN, new JVectorDiskANNMethod());

    private final static Map<SpaceType, Function<Float, Float>> DISTANCE_TRANSLATIONS = ImmutableMap.<
        SpaceType,
        Function<Float, Float>>builder()
        .put(SpaceType.COSINESIMIL, distance -> (2 - distance) / 2)
        .put(SpaceType.INNER_PRODUCT, distance -> distance <= 0 ? 1 / (1 - distance) : distance + 1)
        .build();

    public final static JVector INSTANCE = new JVector(METHODS, Version.LATEST.toString(), DISTANCE_TRANSLATIONS);

    @Override
    public String getExtension() {
        return JVectorFormat.JVECTOR_FILES_SUFFIX;
    }

    @Override
    public String getCompoundExtension() {
        return CUSTOM_COMPOUND_FILE_EXTENSION;
    }

    @Override
    public float score(float rawScore, SpaceType spaceType) {
        return rawScore;
    }

    @Override
    public Float distanceToRadialThreshold(Float distance, SpaceType spaceType) {
        return distance;
    }

    @Override
    public Float scoreToRadialThreshold(Float score, SpaceType spaceType) {
        return score;
    }

    // TODO: Implement this
    @Override
    public ResolvedMethodContext resolveMethod(
        KNNMethodContext knnMethodContext,
        KNNMethodConfigContext knnMethodConfigContext,
        boolean shouldRequireTraining,
        SpaceType spaceType
    ) {
        return methodResolver.resolveMethod(knnMethodContext, knnMethodConfigContext, shouldRequireTraining, spaceType);
    }

    // TODO: add actual file suffix there
    @Override
    public List<String> mmapFileExtensions() {
        return Collections.emptyList();
    }
}
