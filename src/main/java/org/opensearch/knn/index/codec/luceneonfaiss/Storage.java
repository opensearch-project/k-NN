/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.luceneonfaiss;

import lombok.Getter;
import org.apache.lucene.store.IndexInput;

import java.io.IOException;

public class Storage {
    @Getter
    protected long baseOffset;
    @Getter
    protected long sectionSize;

    /**
     * Mark the starting offset and the size of section then skip to the next section.
     *
     * @param input Input read stream.
     * @param singleElementSize Size of atomic element. In file, it only stores the number of elements and the size of element will be
     *                          used to calculate the actual size of section. Ex: size=100, element=int, then the actual section size=400.
     * @throws IOException
     */
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
}
