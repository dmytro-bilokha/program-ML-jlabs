package com.dmytrobilokha.pmljlabs;

public class MnistClassifier {

    // For MNIST we have one label per digit (0-9)
    private static final int NUMBER_OF_LABELS = 10;
    public static void main(String[] cliArgs) {
        if (cliArgs.length != 4) {
            System.err.println("Expected arguments: first - train image file, second - train label file, third - test image, forth - test label");
            System.exit(1);
        }
        var xTrain = readMnistImageFile(cliArgs[0]);
        var yTrain = encodeMnistLabels(readMnistLabelFile(cliArgs[1]));
        var xTest = readMnistImageFile(cliArgs[2]);
        var yTest = readMnistLabelFile(cliArgs[3]);
        train(xTrain, yTrain, xTest, yTest, 200, 1e-5);
    }

    private static double calculateSigmoid(double z) {
        return 1d / (1d + Math.exp(-z));
    }

    private static DoubleMatrix calculateForward(DoubleMatrix x, DoubleMatrix w) {
        return x.multiply(w).apply(MnistClassifier::calculateSigmoid);
    }

    private static int[] classify(DoubleMatrix x, DoubleMatrix w) {
        var yHat = calculateForward(x, w);
        return yHat.indexOfHighestPerRow();
    }

    private static double calculateLoss(DoubleMatrix x, DoubleMatrix y, DoubleMatrix w) {
        var yHat = calculateForward(x, w);
        var logYHat = yHat.apply(Math::log);
        var firstTerm = y.multiplyElements(logYHat);
        var logOneMinYHat = DoubleMatrix.ofOnes(yHat.getRowDimension(), yHat.getColumnDimension())
                .subtract(yHat)
                .apply(Math::log);
        var oneMinY = DoubleMatrix.ofOnes(y.getRowDimension(), y.getColumnDimension()).subtract(y);
        var secondTerm = oneMinY.multiplyElements(logOneMinYHat);
        return -firstTerm.add(secondTerm).sum() / (double) x.getRowDimension();
    }

    private static DoubleMatrix calculateLossGradient(DoubleMatrix x, DoubleMatrix y, DoubleMatrix w) {
        var forward = calculateForward(x, w);
        var gradient = x.transpose().multiply(forward.subtract(y)).scalarMultiply(1d / x.getRowDimension());
        return gradient;
    }

    private static DoubleMatrix train(DoubleMatrix xTrain, DoubleMatrix yTrain, DoubleMatrix xTest, int[] yTest, int iterations, double lr) {
        DoubleMatrix w = DoubleMatrix.ofZeros(xTrain.getColumnDimension(), yTrain.getColumnDimension());
        for (int i = 0; i < iterations; i++) {
            report(i, xTrain, yTrain, xTest, yTest, w);
            var gradient = calculateLossGradient(xTrain, yTrain, w);
            w = w.subtract(gradient.scalarMultiply(lr));
        }
        report(iterations, xTrain, yTrain, xTest, yTest, w);
        return w;
    }

    private static void report(int iteration, DoubleMatrix xTrain, DoubleMatrix yTrain, DoubleMatrix xTest, int[] yTest, DoubleMatrix w) {
        int[] classification = classify(xTest, w);
        int matchesCount = 0;
        for (int i = 0; i < yTest.length; i++) {
            if (classification[i] == yTest[i]) {
                matchesCount++;
            }
        }
        double matchesPercentage = matchesCount * 100d / yTest.length;
        double trainingLoss = calculateLoss(xTrain, yTrain, w);
        System.out.println(iteration + " " + trainingLoss + " " + matchesPercentage);
    }

    private static DoubleMatrix readMnistImageFile(String fileName) {
        byte[] rawFile = FileUtil.readGzippedBinaryFile(fileName);
        int pointer = 0;
        int magicNumber = bytesToInt(rawFile[pointer++], rawFile[pointer++], rawFile[pointer++], rawFile[pointer++]);
        if (magicNumber != 0x0803) {
            throw new RuntimeException("Magic number of a MNIST image file is wrong: " + Integer.toHexString(magicNumber));
        }
        int numberOfImages = bytesToInt(rawFile[pointer++], rawFile[pointer++], rawFile[pointer++], rawFile[pointer++]);
        int rowsInImage = bytesToInt(rawFile[pointer++], rawFile[pointer++], rawFile[pointer++], rawFile[pointer++]);
        int columnsInImage = bytesToInt(rawFile[pointer++], rawFile[pointer++], rawFile[pointer++], rawFile[pointer++]);
        // Each image represents one input row, and one first column contains 1 for bias
        int columnsInMatrix = rowsInImage * columnsInImage + 1;
        double[][] result = new double[numberOfImages][columnsInMatrix];
        for (int rowIndex = 0; rowIndex < numberOfImages; rowIndex++) {
            double[] row = result[rowIndex];
            // Set the first column for bias
            row[0] = 1d;
            for (int columnIndex = 1; columnIndex < columnsInMatrix; columnIndex++, pointer++) {
                row[columnIndex] = Byte.toUnsignedInt(rawFile[pointer]);
            }
        }
        return DoubleMatrix.with2dArray(result);
    }

    private static int[] readMnistLabelFile(String fileName) {
        byte[] rawFile = FileUtil.readGzippedBinaryFile(fileName);
        int pointer = 0;
        int magicNumber = bytesToInt(rawFile[pointer++], rawFile[pointer++], rawFile[pointer++], rawFile[pointer++]);
        if (magicNumber != 0x0801) {
            throw new RuntimeException("Magic number of a MNIST label file is wrong: " + Integer.toHexString(magicNumber));
        }
        int numberOfLabels = bytesToInt(rawFile[pointer++], rawFile[pointer++], rawFile[pointer++], rawFile[pointer++]);
        var output = new int[numberOfLabels];
        for (int labelIndex = 0; labelIndex < numberOfLabels; labelIndex++, pointer++) {
            byte label = rawFile[pointer];
            output[labelIndex] = Byte.toUnsignedInt(label);
        }
        return output;
    }

    private static DoubleMatrix encodeMnistLabels(int[] labelsArray) {
        double[][] labelsData = new double[labelsArray.length][NUMBER_OF_LABELS];
        for (int rowIndex = 0; rowIndex < labelsArray.length; rowIndex++) {
            labelsData[rowIndex][labelsArray[rowIndex]] = 1d;
        }
        return DoubleMatrix.with2dArray(labelsData);
    }

    private static int bytesToInt(byte first, byte second, byte third, byte fourth) {
        return (((first & 0xff) << 24) | ((second & 0xff) << 16) |
                ((third & 0xff) << 8) | (fourth & 0xff));
    }

}
