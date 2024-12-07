package com.dmytrobilokha.pmljlabs;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

public class MultipleRegression {

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
        var w = train(x, y, 100000, 0.001d);
        System.out.println("w=" + w);
        var prediction = calculatePrediction(x, w);
        for (int i = 0; i < 5; i++) {
            System.out.println("X[" + i + "] -> "
                    + " " + prediction.getEntry(i, 0) + " " + y.getEntry(i, 0));
        }
    }

     private static RealMatrix calculatePrediction(RealMatrix x, RealMatrix w) {
        return x.multiply(w);
    }

    private static double calculateLoss(RealMatrix x, RealMatrix y, RealMatrix w) {
        var diff = calculatePrediction(x, w).subtract(y);
        return diff.transpose().multiply(diff).getEntry(0, 0) / (double) x.getRowDimension();
    }

    private static RealMatrix calculateLossGradient(RealMatrix x, RealMatrix y, RealMatrix w) {
        var prediction = calculatePrediction(x, w);
        var gradient = x.transpose().multiply(prediction.subtract(y)).scalarMultiply(2d / x.getRowDimension());
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
