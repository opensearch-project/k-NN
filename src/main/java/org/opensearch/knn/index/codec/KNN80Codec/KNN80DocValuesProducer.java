/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.index.codec.KNN80Codec;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.DocValuesProducer;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import java.io.IOException;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SortedDocValues;
import org.apache.lucene.index.SortedNumericDocValues;
import org.apache.lucene.index.SortedSetDocValues;
import org.opensearch.knn.index.codec.util.KNNCodecUtil;
import org.opensearch.knn.index.codec.util.NativeMemoryCacheKeyHelper;
import org.opensearch.knn.index.memory.NativeMemoryCacheManager;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.opensearch.knn.index.mapper.KNNVectorFieldMapper.KNN_FIELD;

@Log4j2
public class KNN80DocValuesProducer extends DocValuesProducer {
    private final DocValuesProducer delegate;
    private final Map<String, String> fieldNameToVectorFileName = new HashMap<>();

    public KNN80DocValuesProducer(DocValuesProducer delegate, SegmentReadState state) {
        this.delegate = delegate;

        for (FieldInfo field : state.fieldInfos) {
            if (!field.attributes().containsKey(KNN_FIELD)) {
                continue;
            }
            // Only segments that contains BinaryDocValues and doesn't have vector values should be considered.
            // By default, we don't create BinaryDocValues for knn field anymore. However, users can set doc_values = true
            // to create binary doc values explicitly like any other field. Hence, we only want to include fields
            // where approximate search is possible only by BinaryDocValues.
            if (field.getDocValuesType() != DocValuesType.BINARY || field.hasVectorValues()) {
                continue;
            }

            final String vectorIndexFileName = KNNCodecUtil.getEngineFileFromFieldInfo(field, state.segmentInfo);
            if (vectorIndexFileName == null) {
                continue;
            }
            final String cacheKey = NativeMemoryCacheKeyHelper.constructCacheKey(vectorIndexFileName, state.segmentInfo);
            fieldNameToVectorFileName.putIfAbsent(field.getName(), cacheKey);
        }
    }

    @Override
    public BinaryDocValues getBinary(FieldInfo field) throws IOException {
        return delegate.getBinary(field);
    }

    @Override
    public NumericDocValues getNumeric(FieldInfo field) throws IOException {
        return delegate.getNumeric(field);
    }

    @Override
    public SortedDocValues getSorted(FieldInfo field) throws IOException {
        return delegate.getSorted(field);
    }

    @Override
    public SortedNumericDocValues getSortedNumeric(FieldInfo field) throws IOException {
        return delegate.getSortedNumeric(field);
    }

    @Override
    public SortedSetDocValues getSortedSet(FieldInfo field) throws IOException {
        return delegate.getSortedSet(field);
    }

    @Override
    public void checkIntegrity() throws IOException {
        delegate.checkIntegrity();
    }

    @Override
    public void close() throws IOException {
        final NativeMemoryCacheManager nativeMemoryCacheManager = NativeMemoryCacheManager.getInstance();
        fieldNameToVectorFileName.values().forEach(nativeMemoryCacheManager::invalidate);
        delegate.close();
    }

    public final List<String> getOpenedIndexPath() {
        return new ArrayList<>(fieldNameToVectorFileName.values());
    }
}
