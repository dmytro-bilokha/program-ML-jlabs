package com.dmytrobilokha.pmljlabs.enhanced;

import com.dmytrobilokha.pmljlabs.DoubleMatrix;
import com.dmytrobilokha.pmljlabs.FileUtil;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Queue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ConcurrentSkipListSet;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

public class NeuralNetwork {

    // For MNIST we have one label per digit (0-9)
    private static final int NUMBER_OF_LABELS = 10;
    private static final int NUMBER_OF_REPORTERS = 10;
    public static void main(String[] cliArgs) {
        if (cliArgs.length != 5) {
            System.err.println("Expected arguments:");
            System.err.println("1 - train image file");
            System.err.println("2 - train label file");
            System.err.println("3 - test image file");
            System.err.println("4 - test label file");
            System.err.println("5 - report base filename");
            System.exit(1);
        }
        var xTrainRaw = readMnistImageFile(cliArgs[0]);
        var yTrain = encodeMnistLabels(readMnistLabelFile(cliArgs[1]));
        var xTestRaw = readMnistImageFile(cliArgs[2]);
        int[] yTestAll = readMnistLabelFile(cliArgs[3]);
        var reportBaseFilename = cliArgs[4];
        int[] yValidation = Arrays.copyOfRange(yTestAll, 0, yTestAll.length / 2 - 1);
        int[] yTest = Arrays.copyOfRange(yTestAll, yTestAll.length / 2, yTestAll.length - 1);
        var xStandardized = standardizeInput(xTrainRaw, xTestRaw);
        var xTrain = xStandardized.first();
        var xTestPair = splitMatrix(xStandardized.second());
        var xValidation = xTestPair.first();
        var xTest = xTestPair.second();
        var reportingQueue = new LinkedBlockingQueue<ReportingTask>();
        var reportOutputLines = new ConcurrentSkipListSet<ReportingLine>();
        var reporters = new ArrayList<Reporter>();
        for (int i = 0; i < NUMBER_OF_REPORTERS; i++) {
            var reporter = new Reporter(xTrain, yTrain, xValidation, yValidation, reportingQueue, reportOutputLines);
            reporter.start();
            reporters.add(reporter);
        }
        var startMessage = "Starting all at " + LocalDateTime.now();
        System.out.println(startMessage);
        var wPair = train(xTrain, yTrain, reportingQueue, 100, 10, 256, 1d);
        var endTrainingMessage = "Finished training at " + LocalDateTime.now();
        System.out.println(endTrainingMessage);
        dumpMatrixToFile(reportBaseFilename + ".w1", wPair.first());
        dumpMatrixToFlatFile(reportBaseFilename + ".w1.flat", wPair.first());
        dumpMatrixToFile(reportBaseFilename + ".w2", wPair.second());
        dumpMatrixToFlatFile(reportBaseFilename + ".w2.flat", wPair.second());
        // Wait until reporters process all the tasks
        try {
            while (!reportingQueue.isEmpty()) {
                Thread.sleep(10000);
            }
        } catch (InterruptedException e) {
            throw new RuntimeException("Got interrupted while waiting for reporters to empty the queue", e);
        }
        reporters.forEach(Reporter::requestStop);
        try {
            for (var reporter : reporters) {
                reporter.join();
            }
        } catch (InterruptedException e) {
            throw new RuntimeException("Got interrupted while waiting for reporters to stop", e);
        }
        var endReportingMessage = "Finished reporting at " + LocalDateTime.now();
        System.out.println(endReportingMessage);
        FileUtil.writeLinesToFile(reportBaseFilename + ".system",
            List.of(startMessage, endTrainingMessage, endReportingMessage));
        FileUtil.writeLinesToFile(reportBaseFilename + ".lstat",
                reportOutputLines.stream()
                        .map(line -> line.epoch() + " " + line.batch() + " " + line.trainingLoss() + " " + line.matchesPercentage())
                        .toList()
        );
    }

    private static void dumpMatrixToFlatFile(String filePath, DoubleMatrix matrix) {
        FileUtil.writeStringToFile(filePath, matrix.toString(System.lineSeparator(), System.lineSeparator()));
    }

    private static void dumpMatrixToFile(String filePath, DoubleMatrix matrix) {
        FileUtil.writeStringToFile(filePath, matrix.toString(" ", System.lineSeparator()));
    }

    private static double calculateSigmoid(double z) {
        return 1d / (1d + Math.exp(-z));
    }

    private static double calculateReLu(double z) {
        return z <= 0 ? 0d : z;
    }

    private static DoubleMatrix calculateReLuGradient(DoubleMatrix s) {
        // TODO : implement
        return null;
    }

    private static MatrixPair standardizeInput(DoubleMatrix xTrain, DoubleMatrix xTest) {
        double numberOfElements = xTrain.getRowDimension() * xTrain.getColumnDimension();
        double average = xTrain.sum() / numberOfElements;
        var deviationsMatrix = xTrain.scalarAdd(-average);
        double standardDeviation = Math.sqrt(deviationsMatrix.multiplyElements(deviationsMatrix).sum() / numberOfElements);
        var xTrainStandardized = deviationsMatrix.scalarDivide(standardDeviation);
        var xTestStandardized = xTest.scalarAdd(-average).scalarDivide(standardDeviation);
        return new MatrixPair(xTrainStandardized, xTestStandardized);
    }

    private static MatrixPair splitMatrix(DoubleMatrix matrix) {
        int resultingRows = matrix.getRowDimension() / 2;
        var firstMatrix = matrix.getSubMatrix(
                0, resultingRows - 1, 0, matrix.getColumnDimension() - 1);
        var secondMatrix = matrix.getSubMatrix(
                resultingRows, matrix.getRowDimension() - 1, 0, matrix.getColumnDimension() - 1);
        return new MatrixPair(firstMatrix, secondMatrix);
    }

    private static DoubleMatrix calculateSigmoidGradient(DoubleMatrix s) {
        return s.multiplyElements(DoubleMatrix.ofOnesSizedAs(s).subtract(s));
    }

    private static MatrixPair calculateForward(DoubleMatrix x, DoubleMatrix w1, DoubleMatrix w2) {
        var h = x.prependColumn(1d).multiply(w1).apply(NeuralNetwork::calculateSigmoid);
        var yHat = calculateSoftmax(h.prependColumn(1d).multiply(w2));
        return new MatrixPair(yHat, h);
    }

    private static DoubleMatrix calculateSoftmax(DoubleMatrix logits) {
        var exponentials = logits.apply(Math::exp);
        return exponentials.divideRows(exponentials.sumPerRow());
    }

    private static MatrixPair initWeights(int inputVariables, int hiddenNodes, int classes) {
        int w1Rows = inputVariables + 1;
        double w1MaxWeightModule = Math.sqrt(2d / (w1Rows * hiddenNodes));
        DoubleMatrix w1 = DoubleMatrix.ofUniRandoms(w1Rows, hiddenNodes)
                .scalarMultiply(2d * w1MaxWeightModule)
                .scalarAdd(-w1MaxWeightModule);
        int w2Rows = hiddenNodes + 1;
        double w2MaxWeightModule = Math.sqrt(2d / (w2Rows * classes));
        DoubleMatrix w2 = DoubleMatrix.ofUniRandoms(w2Rows, classes)
                .scalarMultiply(2d * w2MaxWeightModule)
                .scalarAdd(-w2MaxWeightModule);
        return new MatrixPair(w1, w2);
    }

    private static int[] classify(DoubleMatrix x, DoubleMatrix w1, DoubleMatrix w2) {
        var yHat = calculateForward(x, w1, w2).first();
        return yHat.indexOfHighestPerRow();
    }

    private static double calculateLoss(DoubleMatrix y, DoubleMatrix yHat) {
        var logYHat = yHat.apply(Math::log);
        return -y.multiplyElements(logYHat).sum() / (double) y.getRowDimension();
    }

    private static MatrixPair calculateBack(DoubleMatrix x, DoubleMatrix y, DoubleMatrix yHat, DoubleMatrix w2, DoubleMatrix h) {
        var yHatMinusY = yHat.subtract(y);
        var w2Gradient = h.prependColumn(1d)
                .transpose()
                .multiply(yHatMinusY)
                .scalarMultiply(1d / x.getRowDimension());
        var w1Gradient = x.prependColumn(1d)
                .transpose()
                .multiply(
                        yHatMinusY.multiply(w2.cutOffFirstRows(1).transpose())
                                .multiplyElements(calculateSigmoidGradient(h))
                )
                .scalarMultiply( 1d / x.getRowDimension());
        return new MatrixPair(w1Gradient, w2Gradient);
    }

    private static MatrixPair train(
            DoubleMatrix xTrain,
            DoubleMatrix yTrain,
            Queue<ReportingTask> reportingQueue,
            int hiddenNodes,
            int epochs,
            int batchSize,
            double lr) {
        int inputVariables = xTrain.getColumnDimension();
        int classes = yTrain.getColumnDimension();
        var initialWeights = initWeights(inputVariables, hiddenNodes, classes);
        DoubleMatrix w1 = initialWeights.first();
        DoubleMatrix w2 = initialWeights.second();
        var xBatches = xTrain.splitRowsInBatches(batchSize);
        var yBatches = yTrain.splitRowsInBatches(batchSize);
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int batch = 0; batch < xBatches.size(); batch++) {
                var xBatch = xBatches.get(batch);
                var yBatch = yBatches.get(batch);
                MatrixPair yHatH = calculateForward(xBatch, w1, w2);
                var yHat = yHatH.first();
                var h = yHatH.second();
                MatrixPair gradients = calculateBack(xBatch, yBatch, yHat, w2, h);
                w1 = w1.subtract(gradients.first().scalarMultiply(lr));
                w2 = w2.subtract(gradients.second().scalarMultiply(lr));
                reportingQueue.add(new ReportingTask(epoch, batch, w1, w2));
            }
        }
        return new MatrixPair(w1, w2);
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
        int columnsInMatrix = rowsInImage * columnsInImage;
        double[][] result = new double[numberOfImages][columnsInMatrix];
        for (int rowIndex = 0; rowIndex < numberOfImages; rowIndex++) {
            double[] row = result[rowIndex];
            for (int columnIndex = 0; columnIndex < columnsInMatrix; columnIndex++, pointer++) {
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

    static class Reporter extends Thread {
        private final DoubleMatrix xTrain;
        private final DoubleMatrix yTrain;
        private final DoubleMatrix xTest;
        private final int[] yTest;
        private final BlockingQueue<ReportingTask> reportingQueue;
        private final Collection<ReportingLine> outputCollection;
        private final AtomicBoolean stopRequested;

        Reporter(
                DoubleMatrix xTrain,
                DoubleMatrix yTrain,
                DoubleMatrix xTest,
                int[] yTest,
                BlockingQueue<ReportingTask> reportingQueue,
                Collection<ReportingLine> outputCollection
        ) {
            this.xTrain = xTrain;
            this.yTrain = yTrain;
            this.xTest = xTest;
            this.yTest = yTest;
            this.reportingQueue = reportingQueue;
            this.outputCollection = outputCollection;
            this.stopRequested = new AtomicBoolean();
        }

        public void requestStop() {
            stopRequested.set(true);
        }

        @Override
        public void run() {
            try {
                while (!stopRequested.get()) {
                    ReportingTask task = reportingQueue.poll(1000, TimeUnit.MILLISECONDS);
                    if (task != null) {
                        report(task.epoch(), task.batch(), task.w1(), task.w2());
                    }
                }
            } catch (InterruptedException e) {
                throw new RuntimeException("Interrupted while waiting for queue", e);
            }
        }

        private void report(int epoch, int batch, DoubleMatrix w1, DoubleMatrix w2) {
            int[] classification = classify(xTest, w1, w2);
            int matchesCount = 0;
            for (int i = 0; i < yTest.length; i++) {
                if (classification[i] == yTest[i]) {
                    matchesCount++;
                }
            }
            double matchesPercentage = matchesCount * 100d / yTest.length;
            var yHatH = calculateForward(xTrain, w1, w2);
            var yHat = yHatH.first();
            double trainingLoss = calculateLoss(yTrain, yHat);
            outputCollection.add(new ReportingLine(epoch,batch, trainingLoss, matchesPercentage));
        }
    }

    record MatrixPair(DoubleMatrix first, DoubleMatrix second) {}
    record ReportingTask(int epoch, int batch, DoubleMatrix w1, DoubleMatrix w2) {}
    record ReportingLine(int epoch, int batch, double trainingLoss, double matchesPercentage) implements Comparable<ReportingLine>{

        @Override
        public int compareTo(ReportingLine o) {
            int result = this.epoch - o.epoch;
            if (result != 0) {
                return result;
            }
            return this.batch - o.batch;
        }

    }

}
