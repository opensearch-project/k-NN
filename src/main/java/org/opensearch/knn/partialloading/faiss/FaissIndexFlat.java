/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.partialloading.faiss;

import lombok.Getter;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.partialloading.search.IdAndDistance;
import org.opensearch.knn.partialloading.search.PartialLoadingSearchParameters;
import org.opensearch.knn.partialloading.storage.Storage;

import java.io.IOException;

public class FaissIndexFlat extends FaissIndex {
    // Maps to IndexFlatL2 e.g. L2 distance
    public static final String IXF2 = "IxF2";
    // Maps to IndexFlatIP e.g. InnerProduct
    public static final String IXFI = "IxFI";

    @Getter
    private final Storage codes = new Storage();
    private int oneVectorByteSize;
    private String indexType;

    public static FaissIndex partiallyLoad(final IndexInput input, final String indexType) throws IOException {
        FaissIndexFlat faissIndexFlat = new FaissIndexFlat();
        readCommonHeader(input, faissIndexFlat);
        faissIndexFlat.oneVectorByteSize = Float.BYTES * faissIndexFlat.getDimension();

        faissIndexFlat.codes.markSection(input, Float.BYTES);
        if (faissIndexFlat.codes.getSectionSize() != (faissIndexFlat.getTotalNumberOfVectors() * faissIndexFlat.oneVectorByteSize)) {
            throw new IllegalStateException("Got an inconsistent bytes size of vector [" + faissIndexFlat.codes.getSectionSize() + "] "
                                                + "when faissIndexFlat.totalNumberOfVectors=" + faissIndexFlat.getTotalNumberOfVectors()
                                                + ", faissIndexFlat.oneVectorByteSize=" + faissIndexFlat.oneVectorByteSize);
        }

        faissIndexFlat.indexType = indexType;

        return faissIndexFlat;
    }

    @Override
    public void search(IndexInput indexInput, IdAndDistance[] results, PartialLoadingSearchParameters searchParameters) {
        throw new UnsupportedOperationException();
    }

    @Override
    public String getIndexType() {
        return indexType;
    }
}
