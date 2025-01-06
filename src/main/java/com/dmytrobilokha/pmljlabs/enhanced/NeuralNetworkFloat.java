package com.dmytrobilokha.pmljlabs.enhanced;

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

public class NeuralNetworkFloat {

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
        int hiddenNodes = 100;
        int epochs = 10;
        int batchSize = 128;
        float lr = 0.25f;
        int reportPeriod = 5;
        var paramsMessage = "Hyperparameters: hiddenNodes=" + hiddenNodes +
                ", epochs=" + epochs + ", batchSize=" + batchSize + ", lr=" + lr + ", reportPeriod=" + reportPeriod;
        System.out.println(paramsMessage);
        var wPair = train(xTrain, yTrain, reportingQueue, hiddenNodes, epochs, batchSize, lr, reportPeriod);
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
            List.of(startMessage, paramsMessage, endTrainingMessage, endReportingMessage));
        FileUtil.writeLinesToFile(reportBaseFilename + ".lstat",
                reportOutputLines.stream()
                        .map(line -> line.epoch() + " " + line.batch() + " " + line.trainingLoss() + " " + line.matchesPercentage())
                        .toList()
        );
    }

    private static void dumpMatrixToFlatFile(String filePath, FloatMatrix matrix) {
        FileUtil.writeStringToFile(filePath, matrix.toString(System.lineSeparator(), System.lineSeparator()));
    }

    private static void dumpMatrixToFile(String filePath, FloatMatrix matrix) {
        FileUtil.writeStringToFile(filePath, matrix.toString(" ", System.lineSeparator()));
    }

    private static float calculateSigmoid(float z) {
        return (float) (1d / (1d + Math.exp(-z)));
    }

    private static float calculateReLu(float z) {
        return z <= 0f ? 0f : z;
    }

    private static FloatMatrix calculateReLuGradient(FloatMatrix s) {
        return s.apply(z -> z <= 0f ? 0f : 1f);
    }

    private static MatrixPair standardizeInput(FloatMatrix xTrain, FloatMatrix xTest) {
        float numberOfElements = xTrain.getRowDimension() * xTrain.getColumnDimension();
        float average = xTrain.sum() / numberOfElements;
        var deviationsMatrix = xTrain.scalarAdd(-average);
        float standardDeviation = (float) Math.sqrt(deviationsMatrix.multiplyElements(deviationsMatrix).sum() / numberOfElements);
        var xTrainStandardized = deviationsMatrix.scalarDivide(standardDeviation);
        var xTestStandardized = xTest.scalarAdd(-average).scalarDivide(standardDeviation);
        return new MatrixPair(xTrainStandardized, xTestStandardized);
    }

    private static MatrixPair splitMatrix(FloatMatrix matrix) {
        int resultingRows = matrix.getRowDimension() / 2;
        var firstMatrix = matrix.getSubMatrix(
                0, resultingRows - 1, 0, matrix.getColumnDimension() - 1);
        var secondMatrix = matrix.getSubMatrix(
                resultingRows, matrix.getRowDimension() - 1, 0, matrix.getColumnDimension() - 1);
        return new MatrixPair(firstMatrix, secondMatrix);
    }

    private static FloatMatrix calculateSigmoidGradient(FloatMatrix s) {
        return s.multiplyElements(FloatMatrix.ofOnesSizedAs(s).subtract(s));
    }

    private static MatrixPair calculateForward(FloatMatrix x, FloatMatrix w1, FloatMatrix w2) {
        var h = x.prependColumn(1f).multiply(w1).apply(NeuralNetworkFloat::calculateReLu);
        var yHat = calculateSoftmax(h.prependColumn(1f).multiply(w2));
        return new MatrixPair(yHat, h);
    }

    private static FloatMatrix calculateSoftmax(FloatMatrix logits) {
        var exponentials = logits.apply((input) -> (float) Math.exp(input));
        return exponentials.divideRows(exponentials.sumPerRow());
    }

    private static MatrixPair initWeights(int inputVariables, int hiddenNodes, int classes) {
        int w1Rows = inputVariables + 1;
        float w1MaxWeightModule = (float) Math.sqrt(2d / (w1Rows * hiddenNodes));
        FloatMatrix w1 = FloatMatrix.ofUniRandoms(w1Rows, hiddenNodes)
                .scalarMultiply(2f * w1MaxWeightModule)
                .scalarAdd(-w1MaxWeightModule);
        int w2Rows = hiddenNodes + 1;
        float w2MaxWeightModule = (float) Math.sqrt(2d / (w2Rows * classes));
        FloatMatrix w2 = FloatMatrix.ofUniRandoms(w2Rows, classes)
                .scalarMultiply(2f * w2MaxWeightModule)
                .scalarAdd(-w2MaxWeightModule);
        return new MatrixPair(w1, w2);
    }

    private static int[] classify(FloatMatrix x, FloatMatrix w1, FloatMatrix w2) {
        var yHat = calculateForward(x, w1, w2).first();
        return yHat.indexOfHighestPerRow();
    }

    private static float calculateLoss(FloatMatrix y, FloatMatrix yHat) {
        var logYHat = yHat.apply((input) -> (float) Math.log(input));
        return -y.multiplyElements(logYHat).sum() / (float) y.getRowDimension();
    }

    private static MatrixPair calculateBack(FloatMatrix x, FloatMatrix y, FloatMatrix yHat, FloatMatrix w2, FloatMatrix h) {
        var yHatMinusY = yHat.subtract(y);
        var w2Gradient = h.prependColumn(1f)
                .transpose()
                .multiply(yHatMinusY)
                .scalarMultiply(1f / x.getRowDimension());
        var w1Gradient = x.prependColumn(1f)
                .transpose()
                .multiply(
                        yHatMinusY.multiply(w2.cutOffFirstRows(1).transpose())
                                .multiplyElements(calculateReLuGradient(h))
                )
                .scalarMultiply( 1f / x.getRowDimension());
        return new MatrixPair(w1Gradient, w2Gradient);
    }

    private static MatrixPair train(
            FloatMatrix xTrain,
            FloatMatrix yTrain,
            Queue<ReportingTask> reportingQueue,
            int hiddenNodes,
            int epochs,
            int batchSize,
            float lr,
            int reportPeriod) {
        int inputVariables = xTrain.getColumnDimension();
        int classes = yTrain.getColumnDimension();
        var initialWeights = initWeights(inputVariables, hiddenNodes, classes);
        FloatMatrix w1 = initialWeights.first();
        FloatMatrix w2 = initialWeights.second();
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
                if (batch % reportPeriod == 0) {
                    reportingQueue.add(new ReportingTask(epoch, batch, w1, w2));
                }
            }
        }
        return new MatrixPair(w1, w2);
    }

    private static FloatMatrix readMnistImageFile(String fileName) {
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
        float[][] result = new float[numberOfImages][columnsInMatrix];
        for (int rowIndex = 0; rowIndex < numberOfImages; rowIndex++) {
            float[] row = result[rowIndex];
            for (int columnIndex = 0; columnIndex < columnsInMatrix; columnIndex++, pointer++) {
                row[columnIndex] = Byte.toUnsignedInt(rawFile[pointer]);
            }
        }
        return FloatMatrix.with2dArray(result);
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

    private static FloatMatrix encodeMnistLabels(int[] labelsArray) {
        float[][] labelsData = new float[labelsArray.length][NUMBER_OF_LABELS];
        for (int rowIndex = 0; rowIndex < labelsArray.length; rowIndex++) {
            labelsData[rowIndex][labelsArray[rowIndex]] = 1f;
        }
        return FloatMatrix.with2dArray(labelsData);
    }

    private static int bytesToInt(byte first, byte second, byte third, byte fourth) {
        return (((first & 0xff) << 24) | ((second & 0xff) << 16) |
                ((third & 0xff) << 8) | (fourth & 0xff));
    }

    static class Reporter extends Thread {
        private final FloatMatrix xTrain;
        private final FloatMatrix yTrain;
        private final FloatMatrix xTest;
        private final int[] yTest;
        private final BlockingQueue<ReportingTask> reportingQueue;
        private final Collection<ReportingLine> outputCollection;
        private final AtomicBoolean stopRequested;

        Reporter(
                FloatMatrix xTrain,
                FloatMatrix yTrain,
                FloatMatrix xTest,
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

        private void report(int epoch, int batch, FloatMatrix w1, FloatMatrix w2) {
            int[] classification = classify(xTest, w1, w2);
            int matchesCount = 0;
            for (int i = 0; i < yTest.length; i++) {
                if (classification[i] == yTest[i]) {
                    matchesCount++;
                }
            }
            float matchesPercentage = matchesCount * 100f / yTest.length;
            var yHatH = calculateForward(xTrain, w1, w2);
            var yHat = yHatH.first();
            float trainingLoss = calculateLoss(yTrain, yHat);
            outputCollection.add(new ReportingLine(epoch,batch, trainingLoss, matchesPercentage));
        }
    }

    record MatrixPair(FloatMatrix first, FloatMatrix second) {}
    record ReportingTask(int epoch, int batch, FloatMatrix w1, FloatMatrix w2) {}
    record ReportingLine(int epoch, int batch, float trainingLoss, float matchesPercentage) implements Comparable<ReportingLine>{

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
