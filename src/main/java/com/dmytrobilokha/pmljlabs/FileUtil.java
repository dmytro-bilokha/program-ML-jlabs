package com.dmytrobilokha.pmljlabs;

import java.io.BufferedInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Stream;
import java.util.zip.GZIPInputStream;

public class FileUtil {

    // Any number of spaces could be a separator
    private static final String TXT_DATA_DELIMITER = "\\p{Zs}+";

    private FileUtil() {
        // Util class
    }

    public static List<String[]> readTxtDataFile(String filePath, int ignoreRows) {
        try (Stream<String> lines = Files.lines(Paths.get(filePath))) {
            return lines
                    .skip(ignoreRows)
                    .map(line -> line.split(TXT_DATA_DELIMITER))
                    .toList();
        } catch (IOException e) {
            throw new RuntimeException("Unable to read data from the file: " + filePath, e);
        }
    }

    public static byte[] readGzippedBinaryFile(String filePath) {
        try (
                var inputStream = Files.newInputStream(Path.of(filePath));
                var gzipInputStream = new GZIPInputStream(inputStream);
        ) {
            return gzipInputStream.readAllBytes();
        } catch (IOException e) {
            throw new RuntimeException("Unable to read data from the file: " + filePath, e);
        }
    }

}
