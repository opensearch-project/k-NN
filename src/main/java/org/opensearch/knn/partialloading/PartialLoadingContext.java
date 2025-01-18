/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.partialloading;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.partialloading.faiss.FaissIndex;
import org.opensearch.knn.partialloading.search.PartialLoadingMode;

import java.io.Closeable;
import java.io.IOException;

@RequiredArgsConstructor
@Getter
public class PartialLoadingContext implements Closeable {
    private final FaissIndex faissIndex;
    private final String vectorFileName;
    private final PartialLoadingMode partialLoadingMode;
    private IndexInput indexInput;

    public synchronized IndexInput getIndexInput(Directory directory) throws IOException {
        if (indexInput != null) {
            return indexInput.clone();
        }
        indexInput = directory.openInput(vectorFileName, IOContext.RANDOM);
        return indexInput.clone();
    }

    @Override
    public void close() throws IOException {
        synchronized (this) {
            if (indexInput != null) {
                indexInput.close();
            }
        }
    }
}
