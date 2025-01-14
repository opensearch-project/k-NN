/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.partialloading.storage;

import lombok.Getter;
import org.apache.lucene.store.IndexInput;

import java.io.IOException;

public class Storage {
    @Getter
    protected long baseOffset;
    @Getter
    protected long sectionSize;

    public void markSection(IndexInput input, int singleElementSize) throws IOException {
        this.sectionSize = input.readLong() * singleElementSize;
        this.baseOffset = input.getFilePointer();
        // Skip the whole section and jump to the next section in the file.
        try {
            input.seek(baseOffset + sectionSize);
        } catch (IOException e) {
            throw new IOException("Failed to partial load where baseOffset=" + baseOffset + ", sectionSize=" + sectionSize, e);
        }
    }

    public int readInt(IndexInput indexInput, final long index) throws IOException {
        indexInput.seek(baseOffset + index * Integer.BYTES);
        return indexInput.readInt();
    }
}
