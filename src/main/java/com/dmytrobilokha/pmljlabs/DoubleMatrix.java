package com.dmytrobilokha.pmljlabs;

import org.apache.commons.math3.exception.NumberIsTooSmallException;
import org.apache.commons.math3.exception.OutOfRangeException;

import java.nio.charset.StandardCharsets;
import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.DoubleSupplier;
import java.util.function.DoubleUnaryOperator;

/**
 *  This implementation was copy-pasted from org.apache.commons.math3.linear.BlockRealMatrix with some changes:
 *  - all methods not used have been removed;
 *  - only one primitive private constructor;
 *  - added factory methods to create a matrix;
 *  - removed not needed checks;
 *  - added some missing, but useful methods (element-by-element multiply, sum, etc.);
 *  - made internal blocks array read-only
 */
public class DoubleMatrix {

    public static final DoubleMatrix NULL = new DoubleMatrix(0, 0, 0, 0, new double[0][0]);
    public static final int BLOCK_SIZE = 52;
    private static final Random randomGenerator = new SecureRandom("Deterministic".getBytes(StandardCharsets.UTF_8));
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
        return new DoubleMatrix(rows, columns, blockRows, blockColumns, createFilledBlocksLayout(rows, columns, () -> 1d));
    }

    public static DoubleMatrix ofSndRandoms(int rows, int columns) {
        ensureCreatableSize(rows, columns);
        int blockRows = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int blockColumns = (columns + BLOCK_SIZE - 1) / BLOCK_SIZE;
        return new DoubleMatrix(rows, columns, blockRows, blockColumns, createFilledBlocksLayout(rows, columns, randomGenerator::nextGaussian));
    }

    public static DoubleMatrix ofZerosSizedAs(DoubleMatrix m) {
        return new DoubleMatrix(m.rows, m.columns, m.blockRows, m.blockColumns, createBlocksLayout(m.rows, m.columns));
    }

    public static DoubleMatrix ofOnesSizedAs(DoubleMatrix m) {
        return new DoubleMatrix(m.rows, m.columns, m.blockRows, m.blockColumns, createFilledBlocksLayout(m.rows, m.columns, () -> 1d));
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

    private static double[][] createFilledBlocksLayout(final int rows, final int columns, DoubleSupplier fillValueSupplier) {
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
                for (int k = 0; k < block.length; k++) {
                    block[k] = fillValueSupplier.getAsDouble();
                }
                blocks[blockIndex] = block;
                ++blockIndex;
            }
        }
        return blocks;
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

    record Coordinates(int row, int column) {}

    private Coordinates calculateInBlocksCoordinates(int row, int column, int columns, int blockColumns) {
        final int iBlock = row / BLOCK_SIZE;
        final int jBlock = column / BLOCK_SIZE;
        final int k = (row - iBlock * BLOCK_SIZE) * calculateBlockWidth(jBlock, columns, blockColumns) +
                (column - jBlock * BLOCK_SIZE);
        return new Coordinates(iBlock * blockColumns + jBlock, k);
    }

    public DoubleMatrix divideRows(DoubleMatrix m) {
        if (this.rows != m.rows || m.columns != 1) {
            throw new IllegalArgumentException("("
                    + this.rows + ", " + this.columns + ") cannot have rows divided by (" + m.rows + ", " + m.columns + ")");
        }
        double[][] outBlocks = createBlocksLayout(rows, columns);
        for (int iBlock = 0; iBlock < blockRows; ++iBlock) {
            final int pStart = iBlock * BLOCK_SIZE;
            final int pEnd = Math.min(pStart + BLOCK_SIZE, rows);
            for (int p = pStart; p < pEnd; ++p) {
                var divisorCoordinates = calculateInBlocksCoordinates(p, 0, 1, 1);
                double divisor = m.blocks[divisorCoordinates.row()][divisorCoordinates.column()];
                for (int jBlock = 0; jBlock < blockColumns; ++jBlock) {
                    final int jWidth = blockWidth(jBlock);
                    final int qStart = jBlock * BLOCK_SIZE;
                    final int qEnd = Math.min(qStart + BLOCK_SIZE, columns);
                    final double[] block = blocks[iBlock * blockColumns + jBlock];
                    final double[] outBlock = outBlocks[iBlock * blockColumns + jBlock];
                    int k = (p - pStart) * jWidth;
                    for (int q = qStart; q < qEnd; ++q) {
                        outBlock[k] = block[k] / divisor;
                        ++k;
                    }
                }
            }
        }
        return new DoubleMatrix(rows, columns, blockRows, blockColumns, outBlocks);
    }

    public DoubleMatrix sumPerRow() {
        double[][] outBlocks = createBlocksLayout(rows, 1);
        for (int iBlock = 0; iBlock < blockRows; ++iBlock) {
            final int pStart = iBlock * BLOCK_SIZE;
            final int pEnd = Math.min(pStart + BLOCK_SIZE, rows);
            for (int p = pStart; p < pEnd; ++p) {
                double rowSum = 0d;
                for (int jBlock = 0; jBlock < blockColumns; ++jBlock) {
                    final int jWidth = blockWidth(jBlock);
                    final int qStart = jBlock * BLOCK_SIZE;
                    final int qEnd = Math.min(qStart + BLOCK_SIZE, columns);
                    final double[] block = blocks[iBlock * blockColumns + jBlock];
                    int k = (p - pStart) * jWidth;
                    for (int q = qStart; q < qEnd; ++q) {
                        rowSum += block[k];
                        ++k;
                    }
                }
                var resultCoordinates = calculateInBlocksCoordinates(p, 0, 1, 1);
                outBlocks[resultCoordinates.row()][resultCoordinates.column()] = rowSum;
            }
        }
        return new DoubleMatrix(rows, 1, blockRows, 1, outBlocks);
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

    public DoubleMatrix prependColumn(double fillValue) {
        int outRows = rows;
        int outColumns = columns + 1;
        int outBlockRows = blockRows;
        int outBlockColumns = (outColumns + BLOCK_SIZE - 1) / BLOCK_SIZE;
        double[][] outBlocks = createBlocksLayout(outRows, outColumns);
        for (int iBlock = 0; iBlock < blockRows; ++iBlock) {
            final int pStart = iBlock * BLOCK_SIZE;
            final int pEnd = Math.min(pStart + BLOCK_SIZE, rows);
            for (int p = pStart; p < pEnd; ++p) {
                var firstEntryInRowCoordinates = calculateInBlocksCoordinates(p, 0, outColumns, outBlockColumns);
                outBlocks[firstEntryInRowCoordinates.row()][firstEntryInRowCoordinates.column()] = fillValue;
                for (int jBlock = 0; jBlock < blockColumns; ++jBlock) {
                    final int jWidth = blockWidth(jBlock);
                    final int qStart = jBlock * BLOCK_SIZE;
                    final int qEnd = Math.min(qStart + BLOCK_SIZE, columns);
                    final double[] block = blocks[iBlock * blockColumns + jBlock];
                    int k = (p - pStart) * jWidth;
                    for (int q = qStart; q < qEnd; ++q) {
                        var resultCoordinates = calculateInBlocksCoordinates(p, q + 1, outColumns, outBlockColumns);
                        outBlocks[resultCoordinates.row()][resultCoordinates.column()] = block[k];
                        ++k;
                    }
                }
            }
        }
        return new DoubleMatrix(outRows, outColumns, outBlockRows, outBlockColumns, outBlocks);
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

    public double[][] getData() {
        final double[][] data = new double[rows][columns];
        final int lastColumns = columns - (blockColumns - 1) * BLOCK_SIZE;
        for (int iBlock = 0; iBlock < blockRows; ++iBlock) {
            final int pStart = iBlock * BLOCK_SIZE;
            final int pEnd = Math.min(pStart + BLOCK_SIZE, rows);
            int regularPos = 0;
            int lastPos = 0;
            for (int p = pStart; p < pEnd; ++p) {
                final double[] dataP = data[p];
                int blockIndex = iBlock * blockColumns;
                int dataPos = 0;
                for (int jBlock = 0; jBlock < blockColumns - 1; ++jBlock) {
                    System.arraycopy(blocks[blockIndex++], regularPos, dataP, dataPos, BLOCK_SIZE);
                    dataPos += BLOCK_SIZE;
                }
                System.arraycopy(blocks[blockIndex], lastPos, dataP, dataPos, lastColumns);
                regularPos += BLOCK_SIZE;
                lastPos    += lastColumns;
            }
        }
        return data;
    }

    public DoubleMatrix cutOffFirstRows(int numberOfRows) {
        return getSubMatrix(numberOfRows, rows - 1, 0, columns - 1);
    }

    public List<DoubleMatrix> splitRowsInBatches(int batchSize) {
       if (batchSize < 1) {
           throw new IllegalArgumentException("Minimum batch size is 1, but got " + batchSize);
       }
       var batches = new ArrayList<DoubleMatrix>();
       int leftoverBatchSize = rows % batchSize;
       for (int startRow = 0, endRow = batchSize - 1; endRow < rows; startRow += batchSize, endRow += batchSize) {
           batches.add(getSubMatrix(startRow, endRow, 0, columns - 1));
       }
       if (leftoverBatchSize > 0) {
           batches.add(getSubMatrix(rows - leftoverBatchSize, rows - 1, 0, columns - 1));
       }
       return batches;
    }

    public DoubleMatrix getSubMatrix(final int startRow, final int endRow,
                                        final int startColumn,
                                        final int endColumn)
            throws OutOfRangeException, NumberIsTooSmallException {
        // safety checks
        if (startRow < 0 || startColumn < 0) {
            throw new IllegalArgumentException("Both start row and start column can not be negative");
        }
        if (endRow >= rows || endColumn >= columns) {
            throw new IllegalArgumentException("Both end row and end column can not excess matrix size");
        }
        if (startRow >= endRow || startColumn >= endColumn) {
            throw new IllegalArgumentException("Start coordinates can not be higher or equal to end coordinates");
        }
        int outRows = endRow - startRow + 1;
        int outColumns = endColumn - startColumn + 1;
        int outBlockRows = (outRows + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int outBlockColumns = (outColumns + BLOCK_SIZE - 1) / BLOCK_SIZE;
        double[][] outBlocks = createBlocksLayout(outRows, outColumns);
        // compute blocks shifts
        final int blockStartRow = startRow / BLOCK_SIZE;
        final int rowsShift = startRow % BLOCK_SIZE;
        final int blockStartColumn = startColumn / BLOCK_SIZE;
        final int columnsShift = startColumn % BLOCK_SIZE;

        // perform extraction block-wise, to ensure good cache behavior
        int pBlock = blockStartRow;
        for (int iBlock = 0; iBlock < outBlockRows; ++iBlock) {
            final int iHeight = calculateBlockHeight(iBlock, outRows, outBlockRows);
            int qBlock = blockStartColumn;
            for (int jBlock = 0; jBlock < outBlockColumns; ++jBlock) {
                final int jWidth = calculateBlockWidth(jBlock, outColumns, outBlockColumns);

                // handle one block of the output matrix
                final int outIndex = iBlock * outBlockColumns + jBlock;
                final double[] outBlock = outBlocks[outIndex];
                final int index = pBlock * blockColumns + qBlock;
                final int width = blockWidth(qBlock);

                final int heightExcess = iHeight + rowsShift - BLOCK_SIZE;
                final int widthExcess = jWidth + columnsShift - BLOCK_SIZE;
                if (heightExcess > 0) {
                    // the submatrix block spans on two blocks rows from the original matrix
                    if (widthExcess > 0) {
                        // the submatrix block spans on two blocks columns from the original matrix
                        final int width2 = blockWidth(qBlock + 1);
                        copyBlockPart(blocks[index], width,
                                rowsShift, BLOCK_SIZE,
                                columnsShift, BLOCK_SIZE,
                                outBlock, jWidth, 0, 0);
                        copyBlockPart(blocks[index + 1], width2,
                                rowsShift, BLOCK_SIZE,
                                0, widthExcess,
                                outBlock, jWidth, 0, jWidth - widthExcess);
                        copyBlockPart(blocks[index + blockColumns], width,
                                0, heightExcess,
                                columnsShift, BLOCK_SIZE,
                                outBlock, jWidth, iHeight - heightExcess, 0);
                        copyBlockPart(blocks[index + blockColumns + 1], width2,
                                0, heightExcess,
                                0, widthExcess,
                                outBlock, jWidth, iHeight - heightExcess, jWidth - widthExcess);
                    } else {
                        // the submatrix block spans on one block column from the original matrix
                        copyBlockPart(blocks[index], width,
                                rowsShift, BLOCK_SIZE,
                                columnsShift, jWidth + columnsShift,
                                outBlock, jWidth, 0, 0);
                        copyBlockPart(blocks[index + blockColumns], width,
                                0, heightExcess,
                                columnsShift, jWidth + columnsShift,
                                outBlock, jWidth, iHeight - heightExcess, 0);
                    }
                } else {
                    // the submatrix block spans on one block row from the original matrix
                    if (widthExcess > 0) {
                        // the submatrix block spans on two blocks columns from the original matrix
                        final int width2 = blockWidth(qBlock + 1);
                        copyBlockPart(blocks[index], width,
                                rowsShift, iHeight + rowsShift,
                                columnsShift, BLOCK_SIZE,
                                outBlock, jWidth, 0, 0);
                        copyBlockPart(blocks[index + 1], width2,
                                rowsShift, iHeight + rowsShift,
                                0, widthExcess,
                                outBlock, jWidth, 0, jWidth - widthExcess);
                    } else {
                        // the submatrix block spans on one block column from the original matrix
                        copyBlockPart(blocks[index], width,
                                rowsShift, iHeight + rowsShift,
                                columnsShift, jWidth + columnsShift,
                                outBlock, jWidth, 0, 0);
                    }
                }
                ++qBlock;
            }
            ++pBlock;
        }
        return new DoubleMatrix(outRows, outColumns, outBlockRows, outBlockColumns, outBlocks);
    }

    private void copyBlockPart(final double[] srcBlock, final int srcWidth,
                               final int srcStartRow, final int srcEndRow,
                               final int srcStartColumn, final int srcEndColumn,
                               final double[] dstBlock, final int dstWidth,
                               final int dstStartRow, final int dstStartColumn) {
        final int length = srcEndColumn - srcStartColumn;
        int srcPos = srcStartRow * srcWidth + srcStartColumn;
        int dstPos = dstStartRow * dstWidth + dstStartColumn;
        for (int srcRow = srcStartRow; srcRow < srcEndRow; ++srcRow) {
            System.arraycopy(srcBlock, srcPos, dstBlock, dstPos, length);
            srcPos += srcWidth;
            dstPos += dstWidth;
        }
    }

    private void ensureSameSize(DoubleMatrix other) {
        if (this.rows != other.rows || this.columns != other.columns) {
            throw new IllegalArgumentException("Matrix sizes are not the same: ("
            + this.rows + ", " + this.columns + ") vs (" + other.rows + ", " + other.columns+ ")");
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

    private int calculateBlockWidth(int blockColumn, int columns, int blockColumns) {
        return (blockColumn == blockColumns - 1) ? columns - blockColumn * BLOCK_SIZE : BLOCK_SIZE;
    }

    private int calculateBlockHeight(int blockRow, int rows, int blockRows) {
        return (blockRow == blockRows - 1) ? rows - blockRow * BLOCK_SIZE : BLOCK_SIZE;
    }

}
