/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.index.util;

import org.opensearch.knn.index.KNNMethod;
import org.opensearch.knn.index.KNNMethodContext;
import org.opensearch.knn.index.SpaceType;

import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.FAISS_NAME;
import static org.opensearch.knn.common.KNNConstants.NMSLIB_NAME;


/**
 * KNNEngine provides the functionality to validate and transform user defined indices into information that can be
 * passed to the respective k-NN library's JNI layer.
 */
public enum KNNEngine implements KNNLibrary {
    NMSLIB(NMSLIB_NAME, Nmslib.INSTANCE),
    FAISS(FAISS_NAME, Faiss.INSTANCE);

    public static final KNNEngine DEFAULT = NMSLIB;

    /**
     * Constructor for KNNEngine
     *
     * @param name name of engine
     * @param knnLibrary library the engine uses
     */
    KNNEngine(String name, KNNLibrary knnLibrary) {
        this.name = name;
        this.knnLibrary = knnLibrary;
    }

    private String name;
    private KNNLibrary knnLibrary;

    /**
     * Get the engine
     *
     * @param name of engine to be fetched
     * @return KNNEngine corresponding to name
     */
    public static KNNEngine getEngine(String name) {
        if (NMSLIB.getName().equalsIgnoreCase(name)){
            return NMSLIB;
        }

        if (FAISS.getName().equals(name)) {
            return FAISS;
        }

        throw new IllegalArgumentException("Invalid engine type: " + name);
    }

    /**
     * Get the engine from the path.
     *
     * @param path to be checked
     * @return KNNEngine corresponding to path
     */
    public static KNNEngine getEngineNameFromPath(String path) {
        if (path.endsWith(KNNEngine.NMSLIB.getExtension())
                || path.endsWith(KNNEngine.NMSLIB.getCompoundExtension())) {
            return KNNEngine.NMSLIB;
        }

        if (path.endsWith(KNNEngine.FAISS.getExtension())
                || path.endsWith(KNNEngine.FAISS.getCompoundExtension())) {
            return KNNEngine.FAISS;
        }

        throw new IllegalArgumentException("No engine matches the path's suffix");
    }

    /**
     * Get the name of the engine
     *
     * @return name of the engine
     */
    public String getName() {
        return name;
    }

    @Override
    public String getLatestBuildVersion() {
        return knnLibrary.getLatestBuildVersion();
    }

    @Override
    public String getLatestLibVersion() {
        return knnLibrary.getLatestLibVersion();
    }

    @Override
    public String getExtension() {
        return knnLibrary.getExtension();
    }

    @Override
    public String getCompoundExtension() {
        return knnLibrary.getCompoundExtension();
    }

    @Override
    public KNNMethod getMethod(String methodName) {
        return knnLibrary.getMethod(methodName);
    }

    @Override
    public float score(float rawScore, SpaceType spaceType) {
        return knnLibrary.score(rawScore, spaceType);
    }

    @Override
    public void validateMethod(KNNMethodContext knnMethodContext) {
        knnLibrary.validateMethod(knnMethodContext);
    }

    @Override
    public Map<String, Object> getMethodAsMap(KNNMethodContext knnMethodContext) {
        return knnLibrary.getMethodAsMap(knnMethodContext);
    }
}
