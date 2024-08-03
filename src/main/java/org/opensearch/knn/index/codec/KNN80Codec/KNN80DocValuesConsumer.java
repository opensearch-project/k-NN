/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN80Codec;

import lombok.NonNull;
import lombok.extern.log4j.Log4j2;
import org.opensearch.common.StopWatch;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.indices.ModelCache;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.codecs.DocValuesConsumer;
import org.apache.lucene.codecs.DocValuesProducer;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexWriter;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.plugin.stats.KNNGraphValue;

import java.io.IOException;

import static org.opensearch.knn.common.KNNConstants.MODEL_ID;

/**
 * This class writes the KNN docvalues to the segments
 */
@Log4j2
class KNN80DocValuesConsumer extends DocValuesConsumer {

    private final Logger logger = LogManager.getLogger(KNN80DocValuesConsumer.class);

    private final DocValuesConsumer delegatee;
    private final SegmentWriteState state;

    KNN80DocValuesConsumer(DocValuesConsumer delegatee, SegmentWriteState state) {
        this.delegatee = delegatee;
        this.state = state;
    }

    @Override
    public void addBinaryField(FieldInfo field, DocValuesProducer valuesProducer) throws IOException {
        delegatee.addBinaryField(field, valuesProducer);
        if (isKNNBinaryFieldRequired(field)) {
            StopWatch stopWatch = new StopWatch();
            stopWatch.start();
            addKNNBinaryField(field, valuesProducer, false, true);
            stopWatch.stop();
            long time_in_millis = stopWatch.totalTime().millis();
            KNNGraphValue.REFRESH_TOTAL_TIME_IN_MILLIS.set(KNNGraphValue.REFRESH_TOTAL_TIME_IN_MILLIS.getValue() + time_in_millis);
            logger.warn("Refresh operation complete in " + time_in_millis + " ms");
        }
    }

    private boolean isKNNBinaryFieldRequired(FieldInfo field) {
        final KNNEngine knnEngine = getKNNEngine(field);
        log.debug(String.format("Read engine [%s] for field [%s]", knnEngine.getName(), field.getName()));
        return field.attributes().containsKey(KNNVectorFieldMapper.KNN_FIELD)
            && KNNEngine.getEnginesThatCreateCustomSegmentFiles().stream().anyMatch(engine -> engine == knnEngine);
    }

    private KNNEngine getKNNEngine(@NonNull FieldInfo field) {
        final String modelId = field.attributes().get(MODEL_ID);
        if (modelId != null) {
            var model = ModelCache.getInstance().get(modelId);
            return model.getModelMetadata().getKnnEngine();
        }
        final String engineName = field.attributes().getOrDefault(KNNConstants.KNN_ENGINE, KNNEngine.DEFAULT.getName());
        return KNNEngine.getEngine(engineName);
    }

    public void addKNNBinaryField(FieldInfo field, DocValuesProducer valuesProducer, boolean isMerge, boolean isRefresh)
        throws IOException {
        NativeIndexWriter.getWriter(field).createKNNIndex(field, valuesProducer, state, isMerge, isRefresh);
    }

    /**
     * Merges in the fields from the readers in mergeState
     *
     * @param mergeState Holds common state used during segment merging
     */
    @Override
    public void merge(MergeState mergeState) {
        try {
            delegatee.merge(mergeState);
            assert mergeState != null;
            assert mergeState.mergeFieldInfos != null;
            for (FieldInfo fieldInfo : mergeState.mergeFieldInfos) {
                DocValuesType type = fieldInfo.getDocValuesType();
                if (type == DocValuesType.BINARY && fieldInfo.attributes().containsKey(KNNVectorFieldMapper.KNN_FIELD)) {
                    StopWatch stopWatch = new StopWatch();
                    stopWatch.start();
                    addKNNBinaryField(fieldInfo, new KNN80DocValuesReader(mergeState), true, false);
                    stopWatch.stop();
                    long time_in_millis = stopWatch.totalTime().millis();
                    KNNGraphValue.MERGE_TOTAL_TIME_IN_MILLIS.set(KNNGraphValue.MERGE_TOTAL_TIME_IN_MILLIS.getValue() + time_in_millis);
                    logger.warn("Merge operation complete in " + time_in_millis + " ms");
                }
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void addSortedSetField(FieldInfo field, DocValuesProducer valuesProducer) throws IOException {
        delegatee.addSortedSetField(field, valuesProducer);
    }

    @Override
    public void addSortedNumericField(FieldInfo field, DocValuesProducer valuesProducer) throws IOException {
        delegatee.addSortedNumericField(field, valuesProducer);
    }

    @Override
    public void addSortedField(FieldInfo field, DocValuesProducer valuesProducer) throws IOException {
        delegatee.addSortedField(field, valuesProducer);
    }

    @Override
    public void addNumericField(FieldInfo field, DocValuesProducer valuesProducer) throws IOException {
        delegatee.addNumericField(field, valuesProducer);
    }

    @Override
    public void close() throws IOException {
        delegatee.close();
    }
}
