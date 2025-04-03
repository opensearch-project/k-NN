/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.util;

import junit.framework.TestCase;
import lombok.SneakyThrows;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.Sort;
import org.apache.lucene.store.Directory;
import org.apache.lucene.util.Version;
import org.junit.Assert;
import org.mockito.Mockito;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.KNN80Codec.KNN80CompoundDirectory;
import org.opensearch.knn.index.codec.KNN80Codec.KNN80CompoundFormat;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;
import org.opensearch.knn.index.vectorvalues.TestVectorValues;

import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.index.codec.util.KNNCodecUtil.calculateArraySize;
import static org.opensearch.knn.index.codec.util.KNNCodecUtil.initializeVectorValues;

public class KNNCodecUtilTests extends TestCase {

    public void testCalculateArraySize() {
        int numVectors = 4;
        int vectorLength = 10;

        // Float data type
        VectorDataType vectorDataType = VectorDataType.FLOAT;
        assertEquals(160, calculateArraySize(numVectors, vectorLength, vectorDataType));

        // Byte data type
        vectorDataType = VectorDataType.BYTE;
        assertEquals(40, calculateArraySize(numVectors, vectorLength, vectorDataType));

        // Binary data type
        vectorDataType = VectorDataType.BINARY;
        assertEquals(40, calculateArraySize(numVectors, vectorLength, vectorDataType));
    }

    public void testGetKNNEnginesLegacyCompoundCodec() throws IOException {
        Codec codec = mock(Codec.class);
        Directory directory = mock(Directory.class);
        SegmentInfo segmentInfo = mock(SegmentInfo.class);
        when(segmentInfo.getVersion()).thenReturn(Version.LUCENE_9_1_0);
        when(segmentInfo.getUseCompoundFile()).thenReturn(true);
        when(segmentInfo.getCodec()).thenReturn(codec);

        KNN80CompoundFormat compoundFormat = mock(KNN80CompoundFormat.class);
        KNN80CompoundDirectory compoundDirectory = mock(KNN80CompoundDirectory.class);
        when(compoundFormat.getCompoundReader(directory, segmentInfo)).thenReturn(compoundDirectory);
        when(codec.compoundFormat()).thenReturn(compoundFormat);
        KNNEngine knnEngine = KNNEngine.FAISS;
        Set<String> SEGMENT_MULTI_FIELD_FILES_FAISS = Set.of("_0.cfe", "_0_2011_long_target_field.faissc", "_0_2011_target_field.faissc");
        when(segmentInfo.files()).thenReturn(SEGMENT_MULTI_FIELD_FILES_FAISS);
        List<String> engineFiles = KNNCodecUtil.getEngineFiles(knnEngine, "target_field", segmentInfo);

        assertEquals(engineFiles.size(), 2);
        assertTrue(engineFiles.get(0).equals("_0_2011_target_field.faissc"));
    }

    public void testGetKNNEngines() throws IOException {
        Codec codec = mock(Codec.class);
        Directory directory = mock(Directory.class);
        SegmentInfo segmentInfo = new SegmentInfo(
            directory,
            Version.LUCENE_10_1_0,
            Version.LUCENE_9_0_0,
            "testSegmentName",
            0,
            true,
            false,
            codec,
            Mockito.mock(Map.class),
            new byte[16],
            Mockito.mock(Map.class),
            Mockito.mock(Sort.class)
        );
        KNN80CompoundFormat compoundFormat = mock(KNN80CompoundFormat.class);
        KNN80CompoundDirectory compoundDirectory = mock(KNN80CompoundDirectory.class);
        when(compoundFormat.getCompoundReader(directory, segmentInfo)).thenReturn(compoundDirectory);
        when(codec.compoundFormat()).thenReturn(compoundFormat);
        KNNEngine knnEngine = KNNEngine.FAISS;
        String[] SEGMENT_MULTI_FIELD_FILES_FAISS = {"_0.cfe", "_0_2011_long_target_field.faiss", "_0_2011_target_field.faiss"};
        when(compoundDirectory.listAll()).thenReturn(SEGMENT_MULTI_FIELD_FILES_FAISS);
        List<String> engineFiles = KNNCodecUtil.getEngineFiles(knnEngine, "target_field", segmentInfo);
        assertEquals(engineFiles.size(), 2);
        assertTrue(engineFiles.get(0).equals("_0_2011_target_field.faiss"));
    }

    @SneakyThrows
    public void testInitializeVectorValues_whenValidVectorValues_thenSuccess() {
        // Give
        final List<float[]> floatArray = List.of(new float[] { 1, 2 }, new float[] { 2, 3 });
        final int dimension = floatArray.get(0).length;
        final TestVectorValues.PreDefinedFloatVectorValues randomVectorValues = new TestVectorValues.PreDefinedFloatVectorValues(
            floatArray
        );
        final KNNVectorValues<float[]> knnVectorValues = KNNVectorValuesFactory.getVectorValues(VectorDataType.FLOAT, randomVectorValues);

        // When
        initializeVectorValues(knnVectorValues);

        // Then
        Assert.assertNotEquals(-1, knnVectorValues.docId());
        Assert.assertArrayEquals(floatArray.get(0), knnVectorValues.getVector(), 0.001f);
        assertEquals(dimension, knnVectorValues.dimension());
    }

    @SneakyThrows
    public void testInitializeVectorValues_whenNoDocs_thenSuccess() {
        // Give
        final List<float[]> floatArray = Collections.emptyList();
        final TestVectorValues.PreDefinedFloatVectorValues randomVectorValues = new TestVectorValues.PreDefinedFloatVectorValues(
            floatArray
        );
        final KNNVectorValues<float[]> knnVectorValues = KNNVectorValuesFactory.getVectorValues(VectorDataType.FLOAT, randomVectorValues);

        // When
        initializeVectorValues(knnVectorValues);
        // Then
        Assert.assertEquals(DocIdSetIterator.NO_MORE_DOCS, knnVectorValues.docId());
    }
}
