/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex;

import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;

import java.io.IOException;

/**
 * Interface which dictates how the index needs to be built
 */
public interface NativeIndexBuildStrategy {

    /**
     * @param indexInfo {@link BuildIndexParams} containing information about the index to be built
     * @throws IOException if an error occurs during the build process
     */
    void buildAndWriteIndex(BuildIndexParams indexInfo) throws IOException;

    /**
     * Build and write index with flush flag. Default implementation calls the non-flag version.
     * RemoteIndexBuildStrategy will override this to use the flag.
     *
     * @param indexInfo {@link BuildIndexParams} containing information about the index to be built
     * @param isFlush Flag indicating if operation is from flush (vs merge)
     * @throws IOException if an error occurs during the build process
     */
    default void buildAndWriteIndex(BuildIndexParams indexInfo, boolean isFlush) throws IOException {
        buildAndWriteIndex(indexInfo);
    }
}
