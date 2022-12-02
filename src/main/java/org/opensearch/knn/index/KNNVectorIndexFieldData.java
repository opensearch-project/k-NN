/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.SortField;
import org.opensearch.common.util.BigArrays;
import org.opensearch.index.fielddata.IndexFieldData;
import org.opensearch.index.fielddata.IndexFieldDataCache;
import org.opensearch.indices.breaker.CircuitBreakerService;
import org.opensearch.search.DocValueFormat;
import org.opensearch.search.MultiValueMode;
import org.opensearch.search.aggregations.support.ValuesSourceType;
import org.opensearch.search.sort.BucketedSort;
import org.opensearch.search.sort.SortOrder;

public class KNNVectorIndexFieldData implements IndexFieldData<KNNVectorDVLeafFieldData> {

    private final String fieldName;
    private final ValuesSourceType valuesSourceType;

    public KNNVectorIndexFieldData(String fieldName, ValuesSourceType valuesSourceType) {
        this.fieldName = fieldName;
        this.valuesSourceType = valuesSourceType;
    }

    @Override
    public String getFieldName() {
        return fieldName;
    }

    @Override
    public ValuesSourceType getValuesSourceType() {
        return valuesSourceType;
    }

    @Override
    public KNNVectorDVLeafFieldData load(LeafReaderContext context) {
        return new KNNVectorDVLeafFieldData(context.reader(), fieldName);
    }

    @Override
    public KNNVectorDVLeafFieldData loadDirect(LeafReaderContext context) {
        return load(context);
    }

    @Override
    public SortField sortField(Object missingValue, MultiValueMode sortMode, XFieldComparatorSource.Nested nested, boolean reverse) {
        throw new UnsupportedOperationException("knn vector field doesn't support this operation");
    }

    @Override
    public BucketedSort newBucketedSort(
        BigArrays bigArrays,
        Object missingValue,
        MultiValueMode sortMode,
        XFieldComparatorSource.Nested nested,
        SortOrder sortOrder,
        DocValueFormat format,
        int bucketSize,
        BucketedSort.ExtraData extra
    ) {
        throw new UnsupportedOperationException("knn vector field doesn't support this operation");
    }

    public static class Builder implements IndexFieldData.Builder {

        private final String name;
        private final ValuesSourceType valuesSourceType;

        public Builder(String name, ValuesSourceType valuesSourceType) {
            this.name = name;
            this.valuesSourceType = valuesSourceType;
        }

        @Override
        public IndexFieldData<?> build(IndexFieldDataCache cache, CircuitBreakerService breakerService) {
            return new KNNVectorIndexFieldData(name, valuesSourceType);
        }
    }
}
