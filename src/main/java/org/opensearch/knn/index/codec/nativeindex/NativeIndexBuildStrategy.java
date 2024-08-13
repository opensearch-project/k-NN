/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex;

import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;

import java.io.IOException;

/**
 * Interface which dictates how the index needs to be built
 */
public interface NativeIndexBuildStrategy {

    void buildAndWriteIndex(BuildIndexParams indexInfo, final KNNVectorValues<?> knnVectorValues) throws IOException;
}
