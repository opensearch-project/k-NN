/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.common.annotation.ExperimentalApi;

/**
 * The build side parameters of a {@link NativeEngineService} operation. The core supplied values are
 * typed fields and a new field can be added without breaking implementors, which method parameters
 * cannot do.
 *
 * <p>Construct through the per operation factory so only the fields that operation defines are set.
 * A field the factory does not take is zero and carries no meaning for that operation; implementations
 * must not read it.
 */
@ExperimentalApi
public record NativeIndexBuildParams(long numDocs, int dim, boolean skipFlat, EngineParameters engineParameters) {

    /** Parameters of {@code initIndex}: the expected document count, the dimension and the engine parameters. */
    public static NativeIndexBuildParams forInit(long numDocs, int dim, EngineParameters engineParameters) {
        return new NativeIndexBuildParams(numDocs, dim, false, engineParameters);
    }

    /** Parameters of {@code insertToIndex} and {@code createIndexFromTemplate}: the dimension and the engine parameters. */
    public static NativeIndexBuildParams forVectors(int dim, EngineParameters engineParameters) {
        return new NativeIndexBuildParams(0, dim, false, engineParameters);
    }

    /** Parameters of {@code writeIndex}: whether the flat vectors are skipped, and the engine parameters. */
    public static NativeIndexBuildParams forWrite(boolean skipFlat, EngineParameters engineParameters) {
        return new NativeIndexBuildParams(0, 0, skipFlat, engineParameters);
    }

    /** Parameters of {@code loadIndex}: the engine parameters only. */
    public static NativeIndexBuildParams forLoad(EngineParameters engineParameters) {
        return new NativeIndexBuildParams(0, 0, false, engineParameters);
    }
}
