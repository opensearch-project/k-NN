/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec;

import org.apache.lucene.codecs.*;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.opensearch.index.mapper.MapperService;

import java.util.function.Function;

/**
 * Wrapper codec for KNN plugin
 *
 * All specific codecs should inherit from this class, this insures that all the relevant format elements are all in the same class
 * and that we are not delegating back and forth between suppliers/consumers of various codec elements from other classes and thus keeps better encapsulation and readability.
 */
public class WrapperCodecForKNNPlugin extends FilterCodec {
    private final String codecName;
    private final Codec codecDelegate;
    private final MapperService mapperService;
    private final PerFieldKnnVectorsFormat perFieldKnnVectorsFormat;
    private final Function<Codec, KNNFormatFacade> knnFormatFacadeSupplier;
    private final KNNFormatFacade knnFormatFacade;

    public WrapperCodecForKNNPlugin(String name, Codec codecDelegate, String codecName, MapperService mapperService, PerFieldKnnVectorsFormat perFieldKnnVectorsFormat, Function<Codec, KNNFormatFacade> knnFormatFacadeSupplier) {
        super(name, codecDelegate);
        this.codecName = codecName;
        this.codecDelegate = codecDelegate;
        this.mapperService = mapperService;
        this.perFieldKnnVectorsFormat = perFieldKnnVectorsFormat;
        this.knnFormatFacadeSupplier = knnFormatFacadeSupplier;
        this.knnFormatFacade = knnFormatFacadeSupplier.apply(codecDelegate);
    }

    @Override
    public DocValuesFormat docValuesFormat() {
        return knnFormatFacade.docValuesFormat();
    }

    @Override
    public CompoundFormat compoundFormat() {
        return knnFormatFacade.compoundFormat();
    }

    @Override
    public KnnVectorsFormat knnVectorsFormat() {
        return perFieldKnnVectorsFormat;
    }

}
