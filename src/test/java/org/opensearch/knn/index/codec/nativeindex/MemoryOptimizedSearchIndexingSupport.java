/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex;

import lombok.experimental.UtilityClass;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;

import java.io.IOException;

@UtilityClass
public class MemoryOptimizedSearchIndexingSupport {
    public static void buildIndex(final BuildIndexParams indexParams) throws IOException {
        MemOptimizedNativeIndexBuildStrategy.getInstance().buildAndWriteIndex(indexParams);
    }
}
