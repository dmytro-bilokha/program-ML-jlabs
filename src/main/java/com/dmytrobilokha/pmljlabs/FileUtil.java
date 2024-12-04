package com.dmytrobilokha.pmljlabs;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Stream;

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

}
