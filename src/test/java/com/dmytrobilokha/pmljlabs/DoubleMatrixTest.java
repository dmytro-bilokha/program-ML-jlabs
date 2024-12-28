package com.dmytrobilokha.pmljlabs;

import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

import java.util.Arrays;
import java.util.stream.Collectors;

@Test(groups = "unit")
public class DoubleMatrixTest {
    private static final double TOLERANCE = 0.00000000001d;

    @DataProvider(name = "generic2dArraysProvider")
    public Object[][] getGeneric2dArrays() {
        return new Object[][]{
                {new double[][]{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}},
                {new double[][]{{1}}},
                {new double[][]{{1, 0}}},
                {new double[][]{{1}, {2}, {3}, {4}, {5}}},
                {new double[][]{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}}},
        };
    }

    @Test(dataProvider = "generic2dArraysProvider")
    public void prependsColumn(double[][] input2dArray) {
        var inputMatrix = DoubleMatrix.with2dArray(input2dArray);
        var resultMatrix = inputMatrix.prependColumn(0.5d);
        var result2dArray = resultMatrix.getData();
        for (int row = 0; row < input2dArray.length; row++) {
            double[] inputRow = input2dArray[row];
            double[] resultRow = result2dArray[row];
            Assert.assertEquals(resultRow[0], 0.5d);
            for (int column = 0; column < inputRow.length; column++) {
                Assert.assertEquals(resultRow[column + 1], inputRow[column]);
            }
        }
    }

    @Test(dataProvider = "generic2dArraysProvider")
    public void calculatesSumPerRow(double[][] input2dArray) {
        var inputMatrix = DoubleMatrix.with2dArray(input2dArray);
        var resultMatrix = inputMatrix.sumPerRow();
        Assert.assertEquals(resultMatrix.getRowDimension(), input2dArray.length);
        Assert.assertEquals(resultMatrix.getColumnDimension(), 1);
        var result2dArray = resultMatrix.getData();
        for (int row = 0; row < input2dArray.length; row++) {
            Assert.assertEquals(result2dArray[row][0], Arrays.stream(input2dArray[row]).sum());
        }
    }

    @Test(dataProvider = "generic2dArraysProvider")
    public void sumPerRowOfDividedBySumEqualsOne(double[][] input2dArray) {
        var inputMatrix = DoubleMatrix.with2dArray(input2dArray);
        var resultMatrix = inputMatrix.divideRows(inputMatrix.sumPerRow()).sumPerRow();
        Assert.assertEquals(resultMatrix.getRowDimension(), input2dArray.length);
        Assert.assertEquals(resultMatrix.getColumnDimension(), 1);
        var result2dArray = resultMatrix.getData();
        for (int row = 0; row < input2dArray.length; row++) {
            Assert.assertTrue(result2dArray[row][0] < 1d + TOLERANCE);
            Assert.assertTrue(result2dArray[row][0] > 1d - TOLERANCE);
        }
    }

    @Test(dataProvider = "generic2dArraysProvider")
    public void dividesRows(double[][] input2dArray) {
        var inputMatrix = DoubleMatrix.with2dArray(input2dArray);
        double[][] divisorArray = new double[input2dArray.length][1];
        for (int i = 0; i < divisorArray.length; i++) {
            divisorArray[i][0] = i + 1;
        }
        var divisorMatrix = DoubleMatrix.with2dArray(divisorArray);
        var resultMatrix = inputMatrix.divideRows(divisorMatrix);
        Assert.assertEquals(resultMatrix.getRowDimension(), input2dArray.length);
        Assert.assertEquals(resultMatrix.getColumnDimension(), inputMatrix.getColumnDimension());
        var result2dArray = resultMatrix.getData();
        for (int row = 0; row < input2dArray.length; row++) {
            double[] inputRowArray  = input2dArray[row];
            double[] resultRowArray = result2dArray[row];
            double divisor = divisorArray[row][0];
            for (int column = 0; column < inputRowArray.length; column++) {
                Assert.assertEquals(resultRowArray[column], inputRowArray[column] / divisor);
            }
        }
    }

    @Test(dataProvider = "generic2dArraysProvider")
    public void exportsToString(double [][] input2dArray) {
        var inputMatrix = DoubleMatrix.with2dArray(input2dArray);
        var expectedString = Arrays.stream(input2dArray)
                .map(Arrays::toString)
                .collect(Collectors.joining(System.lineSeparator()));
        var actualString = "[" + inputMatrix.toString(", " , "]" + System.lineSeparator() + "[") + "]";
        Assert.assertEquals(actualString, expectedString);
    }

    public void splits8rowsInBatches() {
        double[][] input2dArray = new double[][]{
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9},
                {10, 11, 12},
                {13, 14, 15},
                {16, 17, 18},
                {19, 20, 21},
                {22, 23, 24}
        };
        var inputMatrix = DoubleMatrix.with2dArray(input2dArray);
        var batches = inputMatrix.splitRowsInBatches(3);
        Assert.assertEquals(batches.size(), 3);
        var firstBatchMatrix = batches.get(0);
        var firstBatchArray = firstBatchMatrix.getData();
        Assert.assertEquals(firstBatchMatrix.getRowDimension(), 3);
        Assert.assertEquals(firstBatchMatrix.getColumnDimension(), 3);
        Assert.assertEquals(firstBatchArray[0], input2dArray[0]);
        Assert.assertEquals(firstBatchArray[1], input2dArray[1]);
        Assert.assertEquals(firstBatchArray[2], input2dArray[2]);
        var secondBatchMatrix = batches.get(1);
        var secondBatchArray = secondBatchMatrix.getData();
        Assert.assertEquals(secondBatchMatrix.getRowDimension(), 3);
        Assert.assertEquals(secondBatchMatrix.getColumnDimension(), 3);
        Assert.assertEquals(secondBatchArray[0], input2dArray[3]);
        Assert.assertEquals(secondBatchArray[1], input2dArray[4]);
        Assert.assertEquals(secondBatchArray[2], input2dArray[5]);
        var thirdBatchMatrix = batches.get(2);
        var thirdBatchArray = thirdBatchMatrix.getData();
        Assert.assertEquals(thirdBatchMatrix.getRowDimension(), 2);
        Assert.assertEquals(thirdBatchMatrix.getColumnDimension(), 3);
        Assert.assertEquals(thirdBatchArray[0], input2dArray[6]);
        Assert.assertEquals(thirdBatchArray[1], input2dArray[7]);
    }

    public void splits6RowsInBatches() {
        double[][] input2dArray = new double[][]{
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9},
                {10, 11, 12},
                {13, 14, 15},
                {16, 17, 18},
        };
        var inputMatrix = DoubleMatrix.with2dArray(input2dArray);
        var batches = inputMatrix.splitRowsInBatches(3);
        Assert.assertEquals(batches.size(), 2);
        var firstBatchMatrix = batches.get(0);
        var firstBatchArray = firstBatchMatrix.getData();
        Assert.assertEquals(firstBatchMatrix.getRowDimension(), 3);
        Assert.assertEquals(firstBatchMatrix.getColumnDimension(), 3);
        Assert.assertEquals(firstBatchArray[0], input2dArray[0]);
        Assert.assertEquals(firstBatchArray[1], input2dArray[1]);
        Assert.assertEquals(firstBatchArray[2], input2dArray[2]);
        var secondBatchMatrix = batches.get(1);
        var secondBatchArray = secondBatchMatrix.getData();
        Assert.assertEquals(secondBatchMatrix.getRowDimension(), 3);
        Assert.assertEquals(secondBatchMatrix.getColumnDimension(), 3);
        Assert.assertEquals(secondBatchArray[0], input2dArray[3]);
        Assert.assertEquals(secondBatchArray[1], input2dArray[4]);
        Assert.assertEquals(secondBatchArray[2], input2dArray[5]);
    }
}
