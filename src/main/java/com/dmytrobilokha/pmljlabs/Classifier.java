package com.dmytrobilokha.pmljlabs;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.DefaultRealMatrixChangingVisitor;
import org.apache.commons.math3.linear.DefaultRealMatrixPreservingVisitor;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealMatrixChangingVisitor;

public class Classifier {

    public static void main(String[] cliArgs) {
        if (cliArgs.length < 1) {
            System.err.println("CSV filename should be provided");
            System.exit(1);
        }
        var filePath = cliArgs[0];
        var fileLines = FileUtil.readTxtDataFile(filePath, 1);
        var numberOfRows = fileLines.size();
        // last column in the file is Y, but we add one column to X for bias
        var fileNumberOfColumns = fileLines.get(0).length;
        var xNumberOfColumns = fileLines.get(0).length;
        var x = new Array2DRowRealMatrix(numberOfRows, xNumberOfColumns);
        var y = new Array2DRowRealMatrix(numberOfRows, 1);
        for (int i = 0; i < numberOfRows; i++) {
            var fileLine = fileLines.get(i);
            // Y is the last column in the data file
            double[] yRow = new double[]{ Double.parseDouble(fileLine[fileNumberOfColumns - 1]) };
            y.setRow(i, yRow);
            double[] xRow = new double[xNumberOfColumns];
            xRow[0] = 1d; // for bias
            for (int outputIndex = 1, inputIndex = 0; outputIndex < xNumberOfColumns; outputIndex++, inputIndex++) {
                xRow[outputIndex] = Double.parseDouble(fileLine[inputIndex]);
            }
            x.setRow(i, xRow);
        }
        var w = train(x, y, 10000, 0.001d);
        System.out.println("w=" + w);
        var classification = classify(x, w);
        var yDoubles = y.transpose().getData()[0];
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

}
