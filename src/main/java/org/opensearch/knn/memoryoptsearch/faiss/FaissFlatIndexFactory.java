/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import lombok.experimental.UtilityClass;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.index.FieldInfo;
import org.opensearch.knn.common.FieldInfoExtractor;

/**
 * Factory that creates the appropriate {@link FaissIndex} flat storage implementation
 * based on the field's configuration.
 */
@UtilityClass
public class FaissFlatIndexFactory {

    /**
     * Returns a {@link FaissIndex} to use as flat storage for the given field, or {@code null}
     * if the FAISS file's own flat storage should be used.
     */
    public static FaissIndex create(final FieldInfo fieldInfo, final FlatVectorsReader flatVectorsReader) {
        if (FieldInfoExtractor.isFaissBBQ(fieldInfo)) {
            return new FaissBBQFlatIndex(flatVectorsReader, fieldInfo.getName());
        }
        return null;
    }
}
