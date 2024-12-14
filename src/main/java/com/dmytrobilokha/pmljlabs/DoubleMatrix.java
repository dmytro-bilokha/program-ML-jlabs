package com.dmytrobilokha.pmljlabs;

import java.util.Arrays;
import java.util.function.DoubleUnaryOperator;

/**
 *  This implementation was copy-pasted from org.apache.commons.math3.linear.BlockRealMatrix with some changes:
 *  - all methods not used have been removed;
 *  - only one primitive private constructor;
 *  - added factory methods to create a matrix;
 *  - removed not needed checks;
 *  - added some missing, but useful methods (element-by-element multiply, sum, etc.);
 */
public class DoubleMatrix {

    public static final int BLOCK_SIZE = 52;
    private final double[][] blocks;
    private final int rows;
    private final int columns;
    private final int blockRows;
    private final int blockColumns;

    private DoubleMatrix(int rows, int columns, int blockRows, int blockColumns, double[][] blockData) {
        this.rows = rows;
        this.columns = columns;
        this.blockRows = blockRows;
        this.blockColumns = blockColumns;
        blocks = blockData;
    }

    private static double[][] toBlocksLayout(final double[][] rawData) {
        final int rows = rawData.length;
        final int columns = rawData[0].length;
        final int blockRows = (rows    + BLOCK_SIZE - 1) / BLOCK_SIZE;
        final int blockColumns = (columns + BLOCK_SIZE - 1) / BLOCK_SIZE;
        // convert array
        final double[][] blocks = new double[blockRows * blockColumns][];
        int blockIndex = 0;
        for (int iBlock = 0; iBlock < blockRows; ++iBlock) {
            final int pStart = iBlock * BLOCK_SIZE;
            final int pEnd = Math.min(pStart + BLOCK_SIZE, rows);
            final int iHeight = pEnd - pStart;
            for (int jBlock = 0; jBlock < blockColumns; ++jBlock) {
                final int qStart = jBlock * BLOCK_SIZE;
                final int qEnd = Math.min(qStart + BLOCK_SIZE, columns);
                final int jWidth = qEnd - qStart;

                // allocate new block
                final double[] block = new double[iHeight * jWidth];
                blocks[blockIndex] = block;

                // copy data
                int index = 0;
                for (int p = pStart; p < pEnd; ++p) {
                    System.arraycopy(rawData[p], qStart, block, index, jWidth);
                    index += jWidth;
                }
                ++blockIndex;
            }
        }
        return blocks;
    }

    private static double[][] createBlocksLayout(final int rows, final int columns) {
        final int blockRows = (rows    + BLOCK_SIZE - 1) / BLOCK_SIZE;
        final int blockColumns = (columns + BLOCK_SIZE - 1) / BLOCK_SIZE;

        final double[][] blocks = new double[blockRows * blockColumns][];
        int blockIndex = 0;
        for (int iBlock = 0; iBlock < blockRows; ++iBlock) {
            final int pStart = iBlock * BLOCK_SIZE;
            final int pEnd = Math.min(pStart + BLOCK_SIZE, rows);
            final int iHeight = pEnd - pStart;
            for (int jBlock = 0; jBlock < blockColumns; ++jBlock) {
                final int qStart = jBlock * BLOCK_SIZE;
                final int qEnd = Math.min(qStart + BLOCK_SIZE, columns);
                final int jWidth = qEnd - qStart;
                blocks[blockIndex] = new double[iHeight * jWidth];
                ++blockIndex;
            }
        }
        return blocks;
    }

    private static double[][] createFilledBlocksLayout(final int rows, final int columns, double fillValue) {
        final int blockRows = (rows    + BLOCK_SIZE - 1) / BLOCK_SIZE;
        final int blockColumns = (columns + BLOCK_SIZE - 1) / BLOCK_SIZE;

        final double[][] blocks = new double[blockRows * blockColumns][];
        int blockIndex = 0;
        for (int iBlock = 0; iBlock < blockRows; ++iBlock) {
            final int pStart = iBlock * BLOCK_SIZE;
            final int pEnd = Math.min(pStart + BLOCK_SIZE, rows);
            final int iHeight = pEnd - pStart;
            for (int jBlock = 0; jBlock < blockColumns; ++jBlock) {
                final int qStart = jBlock * BLOCK_SIZE;
                final int qEnd = Math.min(qStart + BLOCK_SIZE, columns);
                final int jWidth = qEnd - qStart;
                double[] block = new double[iHeight * jWidth];
                Arrays.fill(block, fillValue);
                blocks[blockIndex] = block;
                ++blockIndex;
            }
        }
        return blocks;
    }

    public static DoubleMatrix ofZeros(int rows, int columns) {
        ensureCreatableSize(rows, columns);
        int blockRows = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int blockColumns = (columns + BLOCK_SIZE - 1) / BLOCK_SIZE;
        return new DoubleMatrix(rows, columns, blockRows, blockColumns, createBlocksLayout(rows, columns));
    }

    public static DoubleMatrix ofOnes(int rows, int columns) {
        ensureCreatableSize(rows, columns);
        int blockRows = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int blockColumns = (columns + BLOCK_SIZE - 1) / BLOCK_SIZE;
        return new DoubleMatrix(rows, columns, blockRows, blockColumns, createFilledBlocksLayout(rows, columns, 1d));
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
        int blockRows = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int blockColumns = (columns + BLOCK_SIZE - 1) / BLOCK_SIZE;
        return new DoubleMatrix(rows, columns, blockRows, blockColumns, toBlocksLayout(array));
    }

    public int getRowDimension() {
        return rows;
    }

    public int getColumnDimension() {
        return columns;
    }

    public DoubleMatrix add(DoubleMatrix m) {
        ensureSameSize(m);
        double[][] outBlocks = createBlocksLayout(rows, columns);
        for (int blockIndex = 0; blockIndex < outBlocks.length; ++blockIndex) {
            final double[] outBlock = outBlocks[blockIndex];
            final double[] tBlock = blocks[blockIndex];
            final double[] mBlock = m.blocks[blockIndex];
            for (int k = 0; k < outBlock.length; ++k) {
                outBlock[k] = tBlock[k] + mBlock[k];
            }
        }
        return new DoubleMatrix(rows, columns, blockRows, blockColumns, outBlocks);
    }

    public DoubleMatrix subtract(DoubleMatrix m) {
        ensureSameSize(m);
        double[][] outBlocks = createBlocksLayout(rows, columns);
        for (int blockIndex = 0; blockIndex < outBlocks.length; ++blockIndex) {
            final double[] outBlock = outBlocks[blockIndex];
            final double[] tBlock = blocks[blockIndex];
            final double[] mBlock = m.blocks[blockIndex];
            for (int k = 0; k < outBlock.length; ++k) {
                outBlock[k] = tBlock[k] - mBlock[k];
            }
        }
        return new DoubleMatrix(rows, columns, blockRows, blockColumns, outBlocks);
    }

    public DoubleMatrix multiplyElements(DoubleMatrix m) {
        ensureSameSize(m);
        double[][] outBlocks = createBlocksLayout(rows, columns);
        for (int blockIndex = 0; blockIndex < outBlocks.length; ++blockIndex) {
            final double[] outBlock = outBlocks[blockIndex];
            final double[] tBlock = blocks[blockIndex];
            final double[] mBlock = m.blocks[blockIndex];
            for (int k = 0; k < outBlock.length; ++k) {
                outBlock[k] = tBlock[k] * mBlock[k];
            }
        }
        return new DoubleMatrix(rows, columns, blockRows, blockColumns, outBlocks);
    }

    public DoubleMatrix scalarMultiply(double s) {
        double[][] outBlocks = createBlocksLayout(rows, columns);
        for (int blockIndex = 0; blockIndex < outBlocks.length; ++blockIndex) {
            final double[] outBlock = outBlocks[blockIndex];
            final double[] tBlock = blocks[blockIndex];
            for (int k = 0; k < outBlock.length; ++k) {
                outBlock[k] = tBlock[k] * s;
            }
        }
        return new DoubleMatrix(rows, columns, blockRows, blockColumns, outBlocks);
    }

    public double sum() {
        double sum = 0;
        for (int blockIndex = 0; blockIndex < blocks.length; ++blockIndex) {
            final double[] tBlock = blocks[blockIndex];
            for (int k = 0; k < tBlock.length; ++k) {
                sum += tBlock[k];
            }
        }
        return sum;
    }

    public DoubleMatrix multiply(DoubleMatrix m) {
        if (columns != m.rows) {
            throw new IllegalArgumentException("This matrix has " + columns + " columns, other has " + m.rows + " rows");
        }
        int outRows = rows;
        int outColumns = m.columns;
        int outBlockRows = (outRows + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int outBlockColumns = (outColumns + BLOCK_SIZE - 1) / BLOCK_SIZE;
        double[][] outBlocks = createBlocksLayout(outRows, outColumns);
        // perform multiplication block-wise, to ensure good cache behavior
        int blockIndex = 0;
        for (int iBlock = 0; iBlock < outBlockRows; ++iBlock) {

            final int pStart = iBlock * BLOCK_SIZE;
            final int pEnd = Math.min(pStart + BLOCK_SIZE, rows);

            for (int jBlock = 0; jBlock < outBlockColumns; ++jBlock) {
                final int jWidth = calculateBlockWidth(jBlock, outColumns, outBlockColumns);
                final int jWidth2 = jWidth  + jWidth;
                final int jWidth3 = jWidth2 + jWidth;
                final int jWidth4 = jWidth3 + jWidth;

                // select current block
                final double[] outBlock = outBlocks[blockIndex];

                // perform multiplication on current block
                for (int kBlock = 0; kBlock < blockColumns; ++kBlock) {
                    final int kWidth = blockWidth(kBlock);
                    final double[] tBlock = blocks[iBlock * blockColumns + kBlock];
                    final double[] mBlock = m.blocks[kBlock * m.blockColumns + jBlock];
                    int k = 0;
                    for (int p = pStart; p < pEnd; ++p) {
                        final int lStart = (p - pStart) * kWidth;
                        final int lEnd = lStart + kWidth;
                        for (int nStart = 0; nStart < jWidth; ++nStart) {
                            double sum = 0;
                            int l = lStart;
                            int n = nStart;
                            while (l < lEnd - 3) {
                                sum += tBlock[l] * mBlock[n] +
                                        tBlock[l + 1] * mBlock[n + jWidth] +
                                        tBlock[l + 2] * mBlock[n + jWidth2] +
                                        tBlock[l + 3] * mBlock[n + jWidth3];
                                l += 4;
                                n += jWidth4;
                            }
                            while (l < lEnd) {
                                sum += tBlock[l++] * mBlock[n];
                                n += jWidth;
                            }
                            outBlock[k] += sum;
                            ++k;
                        }
                    }
                }
                // go to next block
                ++blockIndex;
            }
        }
        return new DoubleMatrix(outRows, outColumns, outBlockRows, outBlockColumns, outBlocks);
    }

    public DoubleMatrix transpose() {
        int outRows = columns;
        int outColumns = rows;
        int outBlockRows = (outRows + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int outBlockColumns = (outColumns + BLOCK_SIZE - 1) / BLOCK_SIZE;
        double[][] outBlocks = createBlocksLayout(outRows, outColumns);
        // perform transpose block-wise, to ensure good cache behavior
        int blockIndex = 0;
        for (int iBlock = 0; iBlock < blockColumns; ++iBlock) {
            for (int jBlock = 0; jBlock < blockRows; ++jBlock) {
                // transpose current block
                final double[] outBlock = outBlocks[blockIndex];
                final double[] tBlock = blocks[jBlock * blockColumns + iBlock];
                final int pStart = iBlock * BLOCK_SIZE;
                final int pEnd = Math.min(pStart + BLOCK_SIZE, columns);
                final int qStart = jBlock * BLOCK_SIZE;
                final int qEnd = Math.min(qStart + BLOCK_SIZE, rows);
                int k = 0;
                for (int p = pStart; p < pEnd; ++p) {
                    final int lInc = pEnd - pStart;
                    int l = p - pStart;
                    for (int q = qStart; q < qEnd; ++q) {
                        outBlock[k] = tBlock[l];
                        ++k;
                        l+= lInc;
                    }
                }
                // go to next block
                ++blockIndex;
            }
        }
        return new DoubleMatrix(outRows, outColumns, outBlockRows, outBlockColumns, outBlocks);
    }

    public DoubleMatrix apply(DoubleUnaryOperator operator) {
        double[][] outBlocks = createBlocksLayout(rows, columns);
        for (int blockIndex = 0; blockIndex < outBlocks.length; ++blockIndex) {
            final double[] outBlock = outBlocks[blockIndex];
            final double[] tBlock = blocks[blockIndex];
            for (int k = 0; k < outBlock.length; ++k) {
                outBlock[k] = operator.applyAsDouble(tBlock[k]);
            }
        }
        return new DoubleMatrix(rows, columns, blockRows, blockColumns, outBlocks);
    }

    public int[] indexOfHighestPerRow() {
        int[] result = new int[rows];
        for (int iBlock = 0; iBlock < blockRows; ++iBlock) {
            final int pStart = iBlock * BLOCK_SIZE;
            final int pEnd = Math.min(pStart + BLOCK_SIZE, rows);
            for (int p = pStart; p < pEnd; ++p) {
                int indexOfHighest = 0;
                double highest = blocks[iBlock * blockColumns][(p - pStart) * blockWidth(0)];
                for (int jBlock = 0; jBlock < blockColumns; ++jBlock) {
                    final int jWidth = blockWidth(jBlock);
                    final int qStart = jBlock * BLOCK_SIZE;
                    final int qEnd = Math.min(qStart + BLOCK_SIZE, columns);
                    final double[] block = blocks[iBlock * blockColumns + jBlock];
                    int k = (p - pStart) * jWidth;
                    for (int q = qStart; q < qEnd; ++q) {
                        double value = block[k];
                        if (value > highest) {
                            highest = value;
                            indexOfHighest = q;
                        }
                        ++k;
                    }
                }
                result[p] = indexOfHighest;
            }
        }
        return result;
    }

    private void ensureSameSize(DoubleMatrix other) {
        if (this.rows != other.rows || this.columns != other.columns) {
            throw new IllegalArgumentException("Matrix sizes are not the same: ("
            + this.rows + ", " + other.rows + ") vs (" + other.rows + ", " + other.columns+ ")");
        }
    }

    private static void ensureCreatableSize(int rows, int columns) {
        if (rows < 1 || columns < 1) {
            throw new IllegalArgumentException("(" + rows + ", " + columns + ") is not a valid size");
        }
    }

    private int blockHeight(final int blockRow) {
        return (blockRow == blockRows - 1) ? rows - blockRow * BLOCK_SIZE : BLOCK_SIZE;
    }

    private int blockWidth(final int blockColumn) {
        return (blockColumn == blockColumns - 1) ? columns - blockColumn * BLOCK_SIZE : BLOCK_SIZE;
    }

    private static int calculateBlockWidth(int blockColumn, int columns, int blockColumns) {
        return (blockColumn == blockColumns - 1) ? columns - blockColumn * BLOCK_SIZE : BLOCK_SIZE;
    }

}
