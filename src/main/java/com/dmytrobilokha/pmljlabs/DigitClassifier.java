package com.dmytrobilokha.pmljlabs;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.DefaultRealMatrixChangingVisitor;
import org.apache.commons.math3.linear.DefaultRealMatrixPreservingVisitor;
import org.apache.commons.math3.linear.RealMatrix;

public class DigitClassifier {

    public static void main(String[] cliArgs) {
        if (cliArgs.length != 4) {
            System.err.println("Expected arguments: first - train image file, second - train label file, third - test image, forth - test label");
            System.exit(1);
        }
        var xTrain = readMnistImageFile(cliArgs[0]);
        var yTrain = readMnistLabelFileOnlyFives(cliArgs[1]);
        var w = train(xTrain, yTrain, 100, 1e-5);
        var xTest = readMnistImageFile(cliArgs[2]);
        var yTest = readMnistLabelFileOnlyFives(cliArgs[3]);
        var classification = classify(xTest, w);
        var yDoubles = yTest.transpose().getData()[0];
        var classificationDoubles = classification.transpose().getData()[0];
        int counterOfCorrect = 0;
        int truePositive = 0;
        int falsePositive = 0;
        int trueNegative = 0;
        int falseNegative = 0;
        for (int i = 0; i < yDoubles.length; i++) {
            var classificationLong = Math.round(classificationDoubles[i]);
            var yLong = Math.round(yDoubles[i]);
            if (classificationLong == yLong) {
                counterOfCorrect++;
                if (classificationLong == 1) {
                    truePositive++;
                } else {
                    trueNegative++;
                }
            } else {
                if (classificationLong == 1) {
                    falsePositive++;
                } else {
                    falseNegative++;
                }

            }
        }
        System.out.println("Accuracy rate is " + counterOfCorrect + " / "
                + yDoubles.length + " -> " + (counterOfCorrect * 100d / yDoubles.length));
        System.out.println("Precision (true positive) rate is " + truePositive + " / "
                + (truePositive + falsePositive) + " -> " + (truePositive * 100d / (truePositive + falsePositive)));
        System.out.println("Recall (true negative) rate is " + trueNegative + " / "
                + (trueNegative + falseNegative) + " -> " + (trueNegative * 100d / (trueNegative + falseNegative)));
        System.out.println();
    }

    private static double calculateSigmoid(double z) {
        return 1d / (1d + Math.exp(-z));
    }

    private static RealMatrix calculateForward(RealMatrix x, RealMatrix w) {
        var result = x.multiply(w);
        result.walkInOptimizedOrder(new DefaultRealMatrixChangingVisitor() {

            @Override
            public double visit(int row, int column, double value) {
                return calculateSigmoid(value);
            }

        });
        return result;
    }

    private static RealMatrix classify(RealMatrix x, RealMatrix w) {
        var result = calculateForward(x, w);
        result.walkInOptimizedOrder(new DefaultRealMatrixChangingVisitor() {

            @Override
            public double visit(int row, int column, double value) {
                return Math.round(value);
            }

        });
        return result;
    }

    private static void logInPlace(RealMatrix in) {
        in.walkInOptimizedOrder(new DefaultRealMatrixChangingVisitor(){

            @Override
            public double visit(int row, int column, double value) {
                return Math.log(value);
            }

        });
    }

    private static double calculateLoss(RealMatrix x, RealMatrix y, RealMatrix w) {
        var yHat = calculateForward(x, w);
        var logYHat = yHat.copy();
        logInPlace(logYHat);
        var firstTerm = y.transpose().multiply(logYHat);
        var logOneMinYHat = yHat.scalarMultiply(-1d).scalarAdd(1d);
        logInPlace(logOneMinYHat);
        var oneMinY = y.scalarMultiply(-1d).scalarAdd(1d);
        var secondTerm = oneMinY.transpose().multiply(logOneMinYHat);
        return -firstTerm.add(secondTerm).walkInOptimizedOrder(new DefaultRealMatrixPreservingVisitor() {
            private double sum = 0;

            @Override
            public void visit(int row, int column, double value) {
                sum += value;
            }

            @Override
            public double end() {
                return sum;
            }

        }) / (double) x.getRowDimension();
    }

    private static RealMatrix calculateLossGradient(RealMatrix x, RealMatrix y, RealMatrix w) {
        var forward = calculateForward(x, w);
        var gradient = x.transpose().multiply(forward.subtract(y)).scalarMultiply(1d / x.getRowDimension());
        return gradient;
    }

    private static RealMatrix train(RealMatrix x, RealMatrix y, int iterations, double lr) {
        RealMatrix w = new Array2DRowRealMatrix(x.getColumnDimension(), 1);
        for (int i = 0; i < iterations; i++) {
            double loss = calculateLoss(x, y, w);
            System.out.println("" + i + " " + loss);
            var gradient = calculateLossGradient(x, y, w);
            w = w.subtract(gradient.scalarMultiply(lr));
        }
        return w;
    }

    private static RealMatrix readMnistImageFile(String fileName) {
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
        var outputMatrix = new Array2DRowRealMatrix(numberOfImages, columnsInMatrix);
        for (int rowIndex = 0; rowIndex < numberOfImages; rowIndex++) {
            double[] row = new double[columnsInMatrix];
            // Set the first column for bias
            outputMatrix.setEntry(rowIndex, 0, 1d);
            for (int columnIndex = 1; columnIndex < columnsInMatrix; columnIndex++, pointer++) {
                 outputMatrix.setEntry(rowIndex, columnIndex, Byte.toUnsignedInt(rawFile[pointer]));
            }
        }
        return outputMatrix;
    }

    private static RealMatrix readMnistLabelFileOnlyFives(String fileName) {
        byte[] rawFile = FileUtil.readGzippedBinaryFile(fileName);
        int pointer = 0;
        int magicNumber = bytesToInt(rawFile[pointer++], rawFile[pointer++], rawFile[pointer++], rawFile[pointer++]);
        if (magicNumber != 0x0801) {
            throw new RuntimeException("Magic number of a MNIST label file is wrong: " + Integer.toHexString(magicNumber));
        }
        int numberOfLabels = bytesToInt(rawFile[pointer++], rawFile[pointer++], rawFile[pointer++], rawFile[pointer++]);
        var outputMatrix = new Array2DRowRealMatrix(numberOfLabels, 1);
        for (int rowIndex = 0; rowIndex < numberOfLabels; rowIndex++, pointer++) {
            byte label = rawFile[pointer];
            double effectiveLabel = label == 5 ? 1d : 0d;
            outputMatrix.setEntry(rowIndex, 0, effectiveLabel);
        }
        return outputMatrix;
    }

    private static int bytesToInt(byte first, byte second, byte third, byte fourth) {
        return (((first & 0xff) << 24) | ((second & 0xff) << 16) |
                ((third & 0xff) << 8) | (fourth & 0xff));
    }

}
