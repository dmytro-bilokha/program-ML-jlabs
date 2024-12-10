package com.dmytrobilokha.pmljlabs;

import java.util.Arrays;
import java.util.function.DoubleUnaryOperator;

public class DoubleMatrix {

    private final double[][] data;
    private final int rows;
    private final int columns;

    private DoubleMatrix(int rows, int columns, double[][] data) {
        this.rows = rows;
        this.columns = columns;
        this.data = data;
    }

    public static DoubleMatrix ofZeros(int rows, int columns) {
        ensureCreatableSize(rows, columns);
        double[][] values = new double[rows][columns];
        return new DoubleMatrix(rows, columns, values);
    }

    public static DoubleMatrix ofOnes(int rows, int columns) {
        ensureCreatableSize(rows, columns);
        double[][] values = new double[rows][columns];
        setEntriesToValue(values, 1d);
        return new DoubleMatrix(rows, columns, values);
    }

    public static DoubleMatrix with2dArray(double[][] array) {
        int rows = array.length;
        if (rows < 1) {
            throw new IllegalArgumentException("2D array must have rows");
        }
        int columns = array[0].length;
        if (columns < 1) {
            throw new IllegalArgumentException("2D array must have columns");
        }
        for (double[] row : array) {
            if (row.length != columns) {
                throw new IllegalArgumentException("All rows should have the same number of columns");
            }
        }
        return new DoubleMatrix(rows, columns, array);
    }

    public int getRowDimension() {
        return rows;
    }

    public int getColumnDimension() {
        return columns;
    }

    public DoubleMatrix add(DoubleMatrix m) {
        ensureSameSize(m);
        double[][] result = new double[rows][columns];
        for (int rowIndex = 0; rowIndex < rows; rowIndex++) {
            double[] thisRow = data[rowIndex];
            double[] mRow = m.data[rowIndex];
            double[] resultRow = result[rowIndex];
            for (int columnIndex = 0; columnIndex < columns; columnIndex++) {
                resultRow[columnIndex] = thisRow[columnIndex] + mRow[columnIndex];
            }
        }
        return new DoubleMatrix(rows, columns, result);
    }

    public DoubleMatrix subtract(DoubleMatrix m) {
        ensureSameSize(m);
        double[][] result = new double[rows][columns];
        for (int rowIndex = 0; rowIndex < rows; rowIndex++) {
            double[] thisRow = data[rowIndex];
            double[] mRow = m.data[rowIndex];
            double[] resultRow = result[rowIndex];
            for (int columnIndex = 0; columnIndex < columns; columnIndex++) {
                resultRow[columnIndex] = thisRow[columnIndex] - mRow[columnIndex];
            }
        }
        return new DoubleMatrix(rows, columns, result);
    }

    public DoubleMatrix multiplyElements(DoubleMatrix m) {
        ensureSameSize(m);
        double[][] result = new double[rows][columns];
        for (int rowIndex = 0; rowIndex < rows; rowIndex++) {
            double[] thisRow = data[rowIndex];
            double[] mRow = m.data[rowIndex];
            double[] resultRow = result[rowIndex];
            for (int columnIndex = 0; columnIndex < columns; columnIndex++) {
                resultRow[columnIndex] = thisRow[columnIndex] * mRow[columnIndex];
            }
        }
        return new DoubleMatrix(rows, columns, result);
    }

    public DoubleMatrix scalarMultiply(double s) {
        double[][] result = new double[rows][columns];
        for (int rowIndex = 0; rowIndex < rows; rowIndex++) {
            double[] thisRow = data[rowIndex];
            double[] resultRow = result[rowIndex];
            for (int columnIndex = 0; columnIndex < columns; columnIndex++) {
                resultRow[columnIndex] = thisRow[columnIndex] * s;
            }
        }
        return new DoubleMatrix(rows, columns, result);
    }

    public double sum() {
        double sum = 0;
        for (int rowIndex = 0; rowIndex < rows; rowIndex++) {
            double[] thisRow = data[rowIndex];
            for (int columnIndex = 0; columnIndex < columns; columnIndex++) {
                sum += thisRow[columnIndex];
            }
        }
        return sum;
    }

    public DoubleMatrix multiply(DoubleMatrix m) {
        if (columns != m.rows) {
            throw new IllegalArgumentException("This matrix has " + columns + " columns, other has " + m.rows + " rows");
        }
        int nRows = this.rows;
        int nCols = m.columns;
        int nSum = this.columns;
        double[][] result = new double[nRows][nCols];
        // Will hold a column of "m".
        double[] mColumn = new double[nSum];
        for (int columnIndex = 0; columnIndex < nCols; columnIndex++) {
            // Copy all elements of column "columnIndex" of "m" so that
            // will be in contiguous memory.
            for (int mRow = 0; mRow < nSum; mRow++) {
                mColumn[mRow] = m.data[mRow][columnIndex];
            }
            for (int rowIndex = 0; rowIndex < nRows; rowIndex++) {
                double[] dataRow = data[rowIndex];
                double sum = 0;
                for (int i = 0; i < nSum; i++) {
                    sum += dataRow[i] * mColumn[i];
                }
                result[rowIndex][columnIndex] = sum;
            }
        }
        return new DoubleMatrix(nRows, nCols, result);
    }

    public DoubleMatrix transpose() {
        double[][] result = new double[columns][rows];
        for (int rowIndex = 0; rowIndex < rows; rowIndex++) {
            for (int columnIndex = 0; columnIndex < columns; columnIndex++) {
                result[columnIndex][rowIndex] = data[rowIndex][columnIndex];
            }
        }
        return new DoubleMatrix(columns, rows, result);
    }

    public DoubleMatrix apply(DoubleUnaryOperator operator) {
        double[][] result = new double[rows][columns];
        for (int rowIndex = 0; rowIndex < rows; rowIndex++) {
            double[] thisRow = data[rowIndex];
            double[] resultRow = result[rowIndex];
            for (int columnIndex = 0; columnIndex < columns; columnIndex++) {
                resultRow[columnIndex] = operator.applyAsDouble(thisRow[columnIndex]);
            }
        }
        return new DoubleMatrix(rows, columns, result);
    }

    public int[] indexOfHighestPerRow() {
        int[] result = new int[rows];
        for (int rowIndex = 0; rowIndex < rows; rowIndex++) {
            double[] row = data[rowIndex];
            int indexOfHighest = 0;
            for (int i = 1; i < columns; i++) {
                if (row[i] > row[indexOfHighest]) {
                    indexOfHighest = i;
                }
            }
            result[rowIndex] = indexOfHighest;
        }
        return result;
    }

    private void ensureSameSize(DoubleMatrix other) {
        if (this.rows != other.rows || this.columns != other.columns) {
            throw new IllegalArgumentException("Matrix sizes are not the same: ("
            + this.rows + ", " + other.rows + ") vs (" + other.rows + ", " + other.columns+ ")");
        }
    }

    private static void setEntriesToValue(double[][] entries, double value) {
        for (int rowIndex = 0; rowIndex < entries.length; rowIndex++) {
            double[] row = entries[rowIndex];
            Arrays.fill(row, value);
        }
    }
    private static void ensureCreatableSize(int rows, int columns) {
        if (rows < 1 || columns < 1) {
            throw new IllegalArgumentException("(" + rows + ", " + columns + ") is not a valid size");
        }
    }

}
