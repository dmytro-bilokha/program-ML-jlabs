package com.dmytrobilokha.pmljlabs;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;

public class LinearRegressionWithBias {

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
        var linearModel = train(x, y, 10000, 0.01d);
        System.out.println("w=" + linearModel.w() + " b=" + linearModel.b());
        System.out.println("Prediction for 20: " + (20d * linearModel.w() + linearModel.b()));
    }

    private static RealVector calculatePrediction(RealVector x, LinearModel model) {
        return x.mapMultiply(model.w()).mapAdd(model.b());
    }
    private static double calculateLoss(RealVector x, RealVector y, LinearModel model) {
        var diff = calculatePrediction(x, model).subtract(y);
        return diff.dotProduct(diff) / (double) x.getDimension();
    }

    private static LinearModel train(RealVector x, RealVector y, int iterations, double lr) {
        double w = 0d;
        double b = 0d;
        for (int i = 0; i < iterations; i++) {
            double loss = calculateLoss(x, y, new LinearModel(w, b));
            System.out.println("" + i + " " + loss);
            if (calculateLoss(x, y, new LinearModel(w + lr, b)) < loss) {
                w += lr;
            } else if (calculateLoss(x, y, new LinearModel(w - lr, b)) < loss) {
                w -= lr;
            } else if (calculateLoss(x, y, new LinearModel(w, b + lr)) < loss) {
                b += lr;
            } else if (calculateLoss(x, y, new LinearModel(w, b - lr)) < loss) {
                b -= lr;
            } else {
                return new LinearModel(w, b);
            }
        }
        throw new IllegalStateException("Failed to converge within " + iterations + " iterations");
    }

    record LinearModel(double w, double b) {}
}
