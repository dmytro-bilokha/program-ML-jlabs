package com.dmytrobilokha.pmljlabs;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;

public class LinearRegression {

    public static void main(String[] cliArgs) {
        if (cliArgs.length < 1) {
            System.err.println("CSV filename should be provided");
            System.exit(1);
        }
        var filePath = cliArgs[0];
        var fileLines = FileUtil.readTxtDataFile(filePath, 1);
        var reservationsArray = new double[fileLines.size()];
        var pizzasArray = new double[fileLines.size()];
        for (int i = 0; i < fileLines.size(); i++) {
            reservationsArray[i] = Double.parseDouble(fileLines.get(i)[0]);
            pizzasArray[i] = Double.parseDouble(fileLines.get(i)[1]);
        }
        var x = new ArrayRealVector(reservationsArray, false);
        var y = new ArrayRealVector(pizzasArray, false);
        double w = train(x, y, 10000, 0.01d);
        System.out.println("w=" + w);
        System.out.println("Prediction for 20: " + 20d * w);
    }

    private static RealVector calculatePrediction(RealVector x, double w) {
        return x.mapMultiply(w);
    }
    private static double calculateLoss(RealVector x, RealVector y, double w) {
        var diff = calculatePrediction(x, w).subtract(y);
        return diff.dotProduct(diff) / (double) x.getDimension();
    }

    private static double train(RealVector x, RealVector y, int iterations, double lr) {
        double w = 0d;
        for (int i = 0; i < iterations; i++) {
            double loss = calculateLoss(x, y, w);
            System.out.println("Iteration " + i + ": loss=" + loss);
            if (calculateLoss(x, y, w + lr) < loss) {
                w += lr;
            } else if (calculateLoss(x, y, w - lr) < loss) {
                w -= lr;
            } else {
                return w;
            }
        }
        throw new IllegalStateException("Failed to converge within " + iterations + " iterations");
    }

}
