/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import lombok.Getter;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.CompoundDirectory;
import org.apache.lucene.codecs.CompoundFormat;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.store.*;
import org.apache.lucene.util.CollectionUtil;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Collections;
import java.util.Map;
import java.util.Set;

// TODO: This needs to be moved under the same package name as the Lucene internal package name for {@link Lucene90CompoundReader}
// this way the internal package constants can be accessed directly and we can avoid duplicating them.
@Log4j2
public class JVectorCompoundFormat extends CompoundFormat {
    private final CompoundFormat delegate;
    static final String DATA_EXTENSION = "cfs";

    /** Extension of compound file entries */
    static final String ENTRIES_EXTENSION = "cfe";

    static final String DATA_CODEC = "Lucene90CompoundData";
    static final String ENTRY_CODEC = "Lucene90CompoundEntries";
    static final int VERSION_START = 0;
    static final int VERSION_CURRENT = VERSION_START;
    public JVectorCompoundFormat(CompoundFormat delegate) {
        super();
        this.delegate = delegate;
    }

    @Override
    public CompoundDirectory getCompoundReader(Directory dir, SegmentInfo si, IOContext context) throws IOException {
        return new JVectorCompoundReader(delegate.getCompoundReader(dir, si, context), dir, si, context);
    }

    @Override
    public void write(Directory dir, SegmentInfo si, IOContext context) throws IOException {
        delegate.write(dir, si, context);
    }

    public static class JVectorCompoundReader extends CompoundDirectory {
        private final CompoundDirectory delegate;
        private final String segmentName;
        private final Map<String, FileEntry> entries;
        @Getter
        private final Path directoryBasePath;
        @Getter
        private final String compoundFileName;
        @Getter
        private final Path compoundFilePath;
        private int version;

        public JVectorCompoundReader(CompoundDirectory delegate, Directory directory, SegmentInfo si, IOContext context) throws IOException {
            this.delegate = delegate;
            this.segmentName = si.name;
            this.directoryBasePath = resolveDirectoryPath(directory);
            String entriesFileName =
                    IndexFileNames.segmentFileName(segmentName, "", JVectorCompoundFormat.ENTRIES_EXTENSION);
                            this.entries = readEntries(si.getId(), directory, entriesFileName);
            this.compoundFileName =
                    IndexFileNames.segmentFileName(segmentName, "", JVectorCompoundFormat.DATA_EXTENSION);
            this.compoundFilePath = directoryBasePath.resolve(compoundFileName);
        }

        @Override
        public void checkIntegrity() throws IOException {
            delegate.checkIntegrity();
        }

        @Override
        public String[] listAll() throws IOException {
            return delegate.listAll();
        }

        @Override
        public long fileLength(String name) throws IOException {
            return delegate.fileLength(name);
        }

        /**
         * Returns the offset of the given file in the compound file.
         */
        public long getOffsetInCompoundFile(String name) {
            FileEntry entry = entries.get(name);
            if (entry == null) {
                throw new IllegalArgumentException("No sub-file with id " + name + " found in compound file");
            }
            return entry.offset;
        }

        @Override
        public IndexInput openInput(String name, IOContext context) throws IOException {
            return delegate.openInput(name, context);
        }

        @Override
        public void close() throws IOException {
            delegate.close();
        }

        @Override
        public Set<String> getPendingDeletions() throws IOException {
            return delegate.getPendingDeletions();
        }

        private Map<String, FileEntry> readEntries(
                byte[] segmentID, Directory dir, String entriesFileName) throws IOException {
            Map<String, FileEntry> mapping = null;
            try (ChecksumIndexInput entriesStream =
                         dir.openChecksumInput(entriesFileName, IOContext.READONCE)) {
                Throwable priorE = null;
                try {
                    version =
                            CodecUtil.checkIndexHeader(
                                    entriesStream,
                                    JVectorCompoundFormat.ENTRY_CODEC,
                                    JVectorCompoundFormat.VERSION_START,
                                    JVectorCompoundFormat.VERSION_CURRENT,
                                    segmentID,
                                    "");

                    mapping = readMapping(entriesStream);

                } catch (Throwable exception) {
                    priorE = exception;
                } finally {
                    CodecUtil.checkFooter(entriesStream, priorE);
                }
            }
            return Collections.unmodifiableMap(mapping);
        }

        private Map<String, FileEntry> readMapping(IndexInput entriesStream) throws IOException {
            final int numEntries = entriesStream.readVInt();
            Map<String, FileEntry> mapping = CollectionUtil.newHashMap(numEntries);
            for (int i = 0; i < numEntries; i++) {
                final FileEntry fileEntry = new FileEntry();
                final String id = segmentName + entriesStream.readString();
                FileEntry previous = mapping.put(id, fileEntry);
                if (previous != null) {
                    throw new CorruptIndexException("Duplicate cfs entry id=" + id + " in CFS ", entriesStream);
                }
                fileEntry.offset = entriesStream.readLong();
                fileEntry.length = entriesStream.readLong();
            }
            return mapping;
        }

        public static final class FileEntry {
            long offset;
            long length;
        }

        private Path resolveDirectoryPath(Directory dir) {
            while (!(dir instanceof FSDirectory)) {
                final String dirType = dir.getClass().getName();
                log.debug("unwrapping dir of type: {} to find path", dirType);
                if (dir instanceof FilterDirectory) {
                    dir = ((FilterDirectory) dir).getDelegate();
                } else {
                    throw new IllegalArgumentException("directory must be FSDirectory or a wrapper around it but instead had type: " + dirType);
                }
            }
            final Path path = ((FSDirectory) dir).getDirectory();
            log.debug("resolved directory path from FSDirectory: {}", path);
            return path;
        }
    }
}
