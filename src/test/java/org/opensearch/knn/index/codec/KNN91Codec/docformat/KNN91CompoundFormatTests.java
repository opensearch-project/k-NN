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

package org.opensearch.knn.index.codec.KNN91Codec.docformat;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.CompoundDirectory;
import org.apache.lucene.codecs.CompoundFormat;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexOutput;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.opensearch.common.util.set.Sets;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.KNN91Codec.KNN91Codec;
import org.opensearch.knn.index.codec.KNNCodecTestUtil;
import org.opensearch.knn.index.util.KNNEngine;

import java.io.IOException;
import java.util.Arrays;
import java.util.Set;

import static org.mockito.Mockito.doNothing;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class KNN91CompoundFormatTests extends KNNTestCase {

    private static Directory directory;
    private static Codec codec;

    @BeforeClass
    public static void setStaticVariables() {
        directory = newFSDirectory(createTempDir());
        codec = new KNN91Codec();
    }

    @AfterClass
    public static void closeStaticVariables() throws IOException {
        directory.close();
    }

    public void testGetCompoundReader() throws IOException {
        CompoundDirectory dir = mock(CompoundDirectory.class);
        CompoundFormat delegate = mock(CompoundFormat.class);
        when(delegate.getCompoundReader(null, null, null)).thenReturn(dir);
        KNN91CompoundFormat knn91CompoundFormat = new KNN91CompoundFormat(delegate);
        assertEquals(dir, knn91CompoundFormat.getCompoundReader(null, null, null));
    }

    public void testWrite() throws IOException {
        // Check that all normal engine files correctly get set to compound extension files after write
        String segmentName = "_test";

        Set<String> segmentFiles = Sets.newHashSet(
            String.format("%s_nmslib1%s", segmentName, KNNEngine.NMSLIB.getExtension()),
            String.format("%s_nmslib2%s", segmentName, KNNEngine.NMSLIB.getExtension()),
            String.format("%s_nmslib3%s", segmentName, KNNEngine.NMSLIB.getExtension()),
            String.format("%s_faiss1%s", segmentName, KNNEngine.FAISS.getExtension()),
            String.format("%s_faiss2%s", segmentName, KNNEngine.FAISS.getExtension()),
            String.format("%s_faiss3%s", segmentName, KNNEngine.FAISS.getExtension())
        );

        SegmentInfo segmentInfo = KNNCodecTestUtil.SegmentInfoBuilder.builder(directory, segmentName, segmentFiles.size(), codec).build();

        for (String name : segmentFiles) {
            IndexOutput indexOutput = directory.createOutput(name, IOContext.DEFAULT);
            indexOutput.close();
        }
        segmentInfo.setFiles(segmentFiles);

        CompoundFormat delegate = mock(CompoundFormat.class);
        doNothing().when(delegate).write(directory, segmentInfo, IOContext.DEFAULT);

        KNN91CompoundFormat knn91CompoundFormat = new KNN91CompoundFormat(delegate);
        knn91CompoundFormat.write(directory, segmentInfo, IOContext.DEFAULT);

        assertTrue(segmentInfo.files().isEmpty());

        Arrays.stream(directory.listAll()).forEach(filename -> {
            try {
                directory.deleteFile(filename);
            } catch (IOException e) {
                fail(String.format("Failed to delete: %s", filename));
            }
        });
    }

}
