package com.dmytrobilokha.pmljlabs;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.Collection;
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

    public static void writeLinesToFile(String filePath, Iterable<String> lines) {
        try (var writer = Files.newBufferedWriter(
                Path.of(filePath),
                StandardOpenOption.WRITE, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {
            for (var line : lines) {
                writer.write(line);
                writer.newLine();
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to write lines to file", e);
        }
    }

    public static void writeStringToFile(String filePath, String data) {
        try (var writer = Files.newBufferedWriter(
                Path.of(filePath),
                StandardOpenOption.WRITE, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {
            writer.write(data);
        } catch (IOException e) {
            throw new RuntimeException("Failed to write string to file", e);
        }
    }

}
