/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import lombok.experimental.UtilityClass;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.memoryoptsearch.faiss.binary.FaissBinaryIndex;
import org.opensearch.knn.memoryoptsearch.FlatVectorsReaderWithFieldName;

import java.io.IOException;

/**
 * Util class being used during partial load each section in Faiss index file.
 * Refer to {@link FaissIndex#doLoad(IndexInput,FlatVectorsReaderWithFieldName)} for more details regarding how it is being used.
 */
@UtilityClass
public class FaissIndexLoadUtils {
    /**
     * Each section of Faiss index must start four leading characters indicating its index type.
     * This is being used to read four bytes, then convert them to String.
     *
     * @param input IndexInput reading bytes from Faiss index.
     * @return A string of index type.
     * @throws IOException
     */
    public static String readIndexType(final IndexInput input) throws IOException {
        final byte[] fourBytes = new byte[4];
        input.readBytes(fourBytes, 0, fourBytes.length);
        return new String(fourBytes);
    }

    /**
     * Util function to convert {@link FaissIndex} to {@link FaissBinaryIndex}.
     * If it fails to cast the given index instance, it will throw {@link IllegalArgumentException}.
     *
     * @param index Target index instance to be casted to {@link FaissBinaryIndex}.
     * @return Casted {@link FaissBinaryIndex}.
     */
    public static FaissBinaryIndex toBinaryIndex(final FaissIndex index) {
        if (index instanceof FaissBinaryIndex binaryIndex) {
            return binaryIndex;
        }

        throw new IllegalArgumentException("Failed to convert [" + index.getIndexType() + "] to FaissBinaryIndex.");
    }
}
