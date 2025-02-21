/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.luceneonfaiss;

import lombok.Getter;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.IndexInput;

import java.io.IOException;
import java.util.Map;
import java.util.function.Supplier;

@Getter
public abstract class FaissIndexFlat extends FaissIndex {
    // Maps to IndexFlatL2 e.g. L2 distance
    public static final String IXF2 = "IxF2";
    // Maps to IndexFlatIP e.g. InnerProduct
    public static final String IXFI = "IxFI";
    private static Map<String, Supplier<FaissIndexFlat>> FLAT_INDEX_SUPPLIERS =
        Map.of(IXF2, FaissIndexFlatL2::new, IXFI, FaissIndexFlatIP::new);

    private final Storage codes = new Storage();
    private int oneVectorByteSize;
    private String indexType;

    public static FaissIndex load(final IndexInput input, final String indexType) throws IOException {
        FaissIndexFlat faissIndexFlat = FLAT_INDEX_SUPPLIERS.getOrDefault(
            indexType,
            () -> {throw new IllegalStateException("Faiss index flat [" + indexType + "] is not supported.");}
        ).get();

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

    public VectorEncoding getVectorEncoding() {
        // We only support float[] at the moment.
        return VectorEncoding.FLOAT32;
    }

    public abstract VectorSimilarityFunction getSimilarityFunction();
}
