/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.util;

import lombok.NonNull;
import lombok.experimental.UtilityClass;
import org.apache.lucene.codecs.lucene95.HasIndexSlice;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.store.IndexInput;

import java.io.IOException;

@UtilityClass
public class WarmupUtil {
    public static void readAll(@NonNull final FloatVectorValues floatVectorValues) throws IOException {
        if (floatVectorValues instanceof HasIndexSlice hasIndexSlice) {
            readAll(hasIndexSlice.getSlice());
            return;
        }
        for (int i = 0; i < floatVectorValues.size(); ++i) {
            floatVectorValues.vectorValue(i);
        }
    }

    public static void readAll(@NonNull final ByteVectorValues byteVectorValues) throws IOException {
        if (byteVectorValues instanceof HasIndexSlice hasIndexSlice) {
            readAll(hasIndexSlice.getSlice());
            return;
        }
        for (int i = 0; i < byteVectorValues.size(); ++i) {
            byteVectorValues.vectorValue(i);
        }
    }

    public static void readAll(@NonNull final IndexInput indexInput) throws IOException {
        indexInput.seek(0);
        for (long left = indexInput.length(); left > 0; --left) {
            indexInput.readByte();
        }
    }
}
