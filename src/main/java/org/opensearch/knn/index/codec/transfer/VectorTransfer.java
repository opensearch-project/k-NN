/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.transfer;

import java.io.Closeable;
import java.io.IOException;

/**
 * An interface to transfer vectors from one memory location to another
 * Class is Closeable to be able to release memory once done
 */
public interface VectorTransfer extends Closeable {

    /**
     * Transfer a batch of vectors from one location to another
     * The batch size here is intended to be constant for multiple transfers so should be encapsulated in the
     * implementation. A new batch size should require another instance
     * @throws IOException
     */
    void transferBatch() throws IOException;

    /**
     * Indicates if there are more vectors to transfer
     * @return
     */
    boolean hasNext();

    /**
     * Gives the docIds for transfered vectors
     * @return
     */
    int[] getTransferredDocsIds();

    /**
     * @return the memory address of the vectors transferred
     */
    long getVectorAddress();
}
