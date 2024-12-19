package com.dmytrobilokha.pmljlabs;

import java.util.Queue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;

public class NeuralNetworkBatched {

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
        var reportingQueue = new LinkedBlockingQueue<ReportingTask>();
        var reporter = new Reporter(xTrain, yTrain, xTest, yTest, reportingQueue);
        reporter.start();
        train(xTrain, yTrain, reportingQueue, 200, 2, 20000, 0.01d);
        reportingQueue.add(new ReportingTask(-1, -1, null, null));
    }

    private static double calculateSigmoid(double z) {
        return 1d / (1d + Math.exp(-z));
    }

    private static DoubleMatrix calculateSigmoidGradient(DoubleMatrix s) {
        return s.multiplyElements(DoubleMatrix.ofOnesSizedAs(s).subtract(s));
    }

    private static MatrixPair calculateForward(DoubleMatrix x, DoubleMatrix w1, DoubleMatrix w2) {
        var h = x.prependColumn(1d).multiply(w1).apply(NeuralNetworkBatched::calculateSigmoid);
        var yHat = calculateSoftmax(h.prependColumn(1d).multiply(w2));
        return new MatrixPair(yHat, h);
    }

    private static DoubleMatrix calculateSoftmax(DoubleMatrix logits) {
        var exponentials = logits.apply(Math::exp);
        return exponentials.divideRows(exponentials.sumPerRow());
    }

    private static MatrixPair initWeights(int inputVariables, int hiddenNodes, int classes) {
        int w1Rows = inputVariables + 1;
        DoubleMatrix w1 = DoubleMatrix.ofSndRandoms(w1Rows, hiddenNodes).scalarMultiply(Math.sqrt(1d / w1Rows));
        int w2Rows = hiddenNodes + 1;
        DoubleMatrix w2 = DoubleMatrix.ofSndRandoms(w2Rows, classes).scalarMultiply(Math.sqrt(1d / w2Rows));
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

        Reporter(DoubleMatrix xTrain, DoubleMatrix yTrain, DoubleMatrix xTest, int[] yTest, BlockingQueue<ReportingTask> reportingQueue) {
            this.xTrain = xTrain;
            this.yTrain = yTrain;
            this.xTest = xTest;
            this.yTest = yTest;
            this.reportingQueue = reportingQueue;
        }

        @Override
        public void run() {
            ReportingTask task;
            try {
                while ((task = reportingQueue.take()).epoch() >= 0) {
                    report(task.epoch(), task.batch(), task.w1(), task.w2());
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
            System.out.println(epoch + " " + batch + " " + trainingLoss + " " + matchesPercentage);
        }
    }

    record MatrixPair(DoubleMatrix first, DoubleMatrix second) {}
    record ReportingTask(int epoch, int batch, DoubleMatrix w1, DoubleMatrix w2) {}

}
