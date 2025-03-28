/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import lombok.Getter;
import org.apache.lucene.store.IndexInput;

import java.io.IOException;

/**
 * This section maps to a section in FAISS index with a starting offset and the section size.
 * A FAISS index file consists of multiple logical sections, each beginning with four bytes indicating an index type. A section may contain
 * a nested section or vector storage, forming a tree structure with a top-level index as the starting point.
 *
 * Ex: FAISS index file
 * +------------+ -> 0
 * +            +
 * +    IxMp    + -> FaissSection(offset=0, section_size=120)
 * +------------+ -> 120
 * +            +
 * +    IHNf    + -> FaissSection(offset=120, section_size=380)
 * +------------+ -> 500
 * +            +
 * +    IxF2    + -> FaissSection(offset=500, section_size=700)
 * +------------+ -> 1200
 *
 */
public class FaissSection {
    @Getter
    private long baseOffset;
    @Getter
    private long sectionSize;

    /**
     * Mark the starting offset and the size of section then skip to the next section.
     *
     * @param input Input read stream.
     * @param singleElementSize Size of atomic element. In file, it only stores the number of elements and the size of element will be
     *                          used to calculate the actual size of section. Ex: size=100, element=int, then the actual section size=400.
     * @throws IOException
     */
    public FaissSection(IndexInput input, int singleElementSize) throws IOException {
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
