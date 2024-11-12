/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import lombok.AllArgsConstructor;
import lombok.Getter;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.RamUsageEstimator;
import org.opensearch.knn.common.FieldInfoExtractor;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.util.KNNCodecUtil;
import org.opensearch.knn.index.engine.KNNEngine;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import static org.opensearch.knn.index.engine.KNNEngine.FAISS;

/**
 * There is 3 Index in one faiss file <-id-><-hnsw-><-Storage->
 * File Structure like followings:
 * |-typeIDMap-||-id_header-|
 *     |-typeHnsw-||-hnsw_header-||-hnswGraph-|
 *         |-typeStorage-||-storage_Header-||-storageVector-|
 * |-idmap_vector-|
 *
 * header would like:
 * |dim|ntotal|dummy|dummy|is_trained|metric_type|metric_arg|
 *
 * Example for HNSW32,Flat:
 * |idMapType|idMapHeader|hnswType|hnswHeader|hnswGraph|flatType|flatHeader|Vectors|IdVector|FOOTER_MAGIC+CHECKSUM|
 */
@Getter
public class FaissEngineFlatKnnVectorsReader extends FaissEngineKnnVectorsReader {

    // 1. A Footer magic number (int - 4 bytes)
    // 2. A checksum algorithm id (int - 4 bytes)
    // 3. A checksum (long - bytes)
    // The checksum is computed on all the bytes written to the file up to that point.
    // Logic where footer is written in Lucene can be found here:
    // https://github.com/apache/lucene/blob/branch_9_0/lucene/core/src/java/org/apache/lucene/codecs/CodecUtil.java#L390-L412
    public static final int FOOT_MAGIC_SIZE = RamUsageEstimator.primitiveSizes.get(Integer.TYPE);
    public static final int ALGORITHM_SIZE = RamUsageEstimator.primitiveSizes.get(Integer.TYPE);
    public static final int CHECKSUM_SIZE = RamUsageEstimator.primitiveSizes.get(Long.TYPE);
    public static final int FLOAT_SIZE = RamUsageEstimator.primitiveSizes.get(Float.TYPE);
    public static final int SIZET_SIZE = RamUsageEstimator.primitiveSizes.get(Long.TYPE);
    public static final int FOOTER_SIZE = FOOT_MAGIC_SIZE + ALGORITHM_SIZE + CHECKSUM_SIZE;

    private Map<String, IndexInput> fieldFileMap;
    private Map<String, MetaInfo> fieldMetaMap;

    @Override
    public void checkIntegrity() throws IOException {

    }

    public FaissEngineFlatKnnVectorsReader(SegmentReadState state) throws IOException {
        fieldFileMap = new HashMap<>();
        fieldMetaMap = new HashMap<>();
        boolean success = false;
        try {
            for (FieldInfo field : state.fieldInfos) {

                KNNEngine knnEngine = KNNCodecUtil.getNativeKNNEngine(field);
                if (knnEngine == null || FAISS != knnEngine) {
                    continue;
                }
                final String vectorIndexFileName = KNNCodecUtil.getNativeEngineFileFromFieldInfo(field, state.segmentInfo);
                if (vectorIndexFileName == null) {
                    continue;
                }
                // TODO for fp16, pq
                VectorDataType vectorDataType = FieldInfoExtractor.extractVectorDataType(field);
                SpaceType spaceType = FieldInfoExtractor.getSpaceType(null, field);
                if (vectorDataType != VectorDataType.FLOAT) {
                    continue;
                }
                String parameter = FieldInfoExtractor.getParameters(field);
                if (parameter == null || parameter.contains("BHNSW")) {
                    continue;
                }
                // TODO if not exist file, change to lucene flatVector
                IndexInput in = state.directory.openInput(vectorIndexFileName, state.context.withRandomAccess());
                if (in == null) {
                    continue;
                }
                fieldFileMap.put(field.getName(), in);
            }
            success = true;
        } finally {
            if (success == false) {
                IOUtils.closeWhileHandlingException(this);
            }
        }

        for (Map.Entry<String, IndexInput> entry : fieldFileMap.entrySet()) {
            IndexInput in = entry.getValue();
            int h = in.readInt();
            MetaInfo metaInfo = read_index_header(in);
            fieldMetaMap.put(entry.getKey(), metaInfo);
        }
    }

    @Override
    public FloatVectorValues getFloatVectorValues(String field) throws IOException {
        MetaInfo metaInfo = fieldMetaMap.get(field);
        IndexInput input = fieldFileMap.get(field);
        FaissEngineFlatVectorValues vectorValues = new FaissEngineFlatVectorValues(metaInfo, input);
        return vectorValues;
    }

    @Override
    public ByteVectorValues getByteVectorValues(String field) throws IOException {
        return null;
    }

    @Override
    public boolean isNativeVectors(String field) {
        return fieldFileMap.containsKey(field) && fieldMetaMap.containsKey(field);
    }

    private MetaInfo read_index_header(IndexInput in) throws IOException {

        int d = in.readInt();
        long ntotal = in.readLong();
        long dummy;
        dummy = in.readLong();
        dummy = in.readLong();
        byte is_trained = in.readByte();
        //
        int metric_type = in.readInt();
        float metric_arg = 0;
        if (metric_type > 1) {
            metric_arg = Float.intBitsToFloat(in.readInt());
        }
        long filesize = in.length();
        // There is (ntotal+1) * idx_t and FOOTER_SIZE
        long idSeek = filesize - (ntotal + 1) * SIZET_SIZE - FOOTER_SIZE;
        // in.seek(idSeek);
        // long size = in.readLong();

        // long[] ids = new long[(int) ntotal];
        // in.readLongs(ids, 0, (int) ntotal);
        long vectorSeek = idSeek - (FLOAT_SIZE * d) * ntotal - SIZET_SIZE;
        // in.seek(vectorSeek);

        // float[] v = new float[(int) (d * ntotal)];
        // size = in.readLong();
        // System.out.println("Vector Size: " + size + " d * ntotal" + d * ntotal);
        // for(int i = 0; i < ntotal; i++) {
        // in.readFloats(v, i * d, d);
        // System.out.println("vector:");
        // for (int j = 0; j < d; j++) {
        // System.out.println(v[i*d + j]);
        // }
        // }
        return new MetaInfo(d, ntotal, is_trained, metric_type, metric_arg, idSeek, vectorSeek);
    }

    @Override
    public void close() throws IOException {
        for (Map.Entry<String, IndexInput> entry : fieldFileMap.entrySet()) {
            IndexInput in = entry.getValue();
            IOUtils.close(in);
        }
    }

    @AllArgsConstructor
    @Getter
    public class MetaInfo {
        int d;
        long ntotal;
        byte isTrained;
        int metricType;
        float metricArg;
        long idSeek;
        long vectorSeek;
    }
}
