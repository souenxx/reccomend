package batch;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import it.unimi.dsi.fastutil.Hash;
import net.librec.common.LibrecException;
import net.librec.conf.Configuration;
import net.librec.math.algorithm.Randoms;
import net.librec.math.structure.DenseMatrix;
import net.librec.math.structure.SequentialAccessSparseMatrix;

import java.util.*;

public class BatchSet {
    protected Configuration conf;
    protected Set<Integer> leftUnion;
    protected Set<Integer> leftNegUnion;
    protected Set<Integer> rightUnion;
    protected Set<Integer> rightNegUnion;
    protected Map<Integer, Set<Integer>> leftToRightSamples;
    protected Map<Integer, Set<Integer>> leftToRightNegSamples;
    protected Map<Integer, Set<Integer>> negRightToLeftSamples;
    protected Map<Integer, Set<Integer>> rightToLeftSamples;
    protected Map<Integer, Set<Integer>> rightToLeftNegSamples;
    protected Table<Integer, Integer, Integer> leftNegRightPosRightTable;
    protected Table<Integer, Integer, Integer> leftPosRightNegRightTable;
    protected Table<Integer, Integer, Integer> rightNegLeftPosLeftTable;
    protected Table<Integer, Integer, Integer> rightPosLeftNegLeftTable;
    protected List<Set<Integer>> trainLeftToRightSet;
    protected List<Set<Integer>> trainRightToLeftSet;
    protected int dimensions;
    protected SequentialAccessSparseMatrix trainSet;
    protected int numLeft;
    protected int numRight;
    protected int batchSize;

    public BatchSet(Configuration conf, SequentialAccessSparseMatrix trainMatrix) {
        this.conf = conf;
        trainSet = trainMatrix;
        numLeft = trainMatrix.rowSize();
        numRight = trainMatrix.columnSize();
    }
    public void sampling() throws LibrecException{
        setup();

        int sampleSize = 0;
        while (sampleSize < this.batchSize) {
           int leftIdx = Randoms.uniform(numLeft);
           Set<Integer> rightSet = trainLeftToRightSet.get(leftIdx);
           if (rightSet.size() == 0 || rightSet.size() == numRight)
               continue;
           int[] rightIndices = trainSet.row(leftIdx).getIndices();
           int rightIdx = rightIndices[Randoms.uniform(rightIndices.length)];

           if (!leftToRightSamples.containsKey(leftIdx)) {
               Set<Integer> rightSamples = new HashSet<>();
               rightSamples.add(rightIdx);
               leftToRightSamples.put(leftIdx, rightSamples);
           } else if (!leftToRightSamples.get(leftIdx).contains(rightIdx)) {
               Set<Integer> rightSamples = leftToRightSamples.get(leftIdx);
               rightSamples.add(rightIdx);
           } else {
               continue;
           }

           if (!rightToLeftSamples.containsKey(rightIdx)) {
               Set<Integer> leftSamples = new HashSet<>();
               leftSamples.add(leftIdx);
               rightToLeftSamples.put(rightIdx, leftSamples);
           } else {
               Set<Integer> leftSamples = rightToLeftSamples.get(rightIdx);
               leftSamples.add(leftIdx);
           }
           leftUnion.add(leftIdx);
           rightUnion.add(rightIdx);
           sampleSize++;
        }
    }
    public void sampling(int batchSize) throws LibrecException{
        this.batchSize = batchSize;
        if (leftToRightSamples != null || rightToLeftSamples != null) {
            leftToRightSamples.clear();
            rightToLeftSamples.clear();
        }
        sampling();

    }

    public void rightNegSampling(int batchSize) throws LibrecException{
        if (leftToRightSamples.size() == 0) {
            sampling(batchSize);
        }
        if (leftToRightNegSamples.size() != 0) {
            leftToRightNegSamples.clear();
            leftPosRightNegRightTable.clear();
            leftNegRightPosRightTable.clear();
            negRightToLeftSamples.clear();
        }
        if (leftToRightNegSamples.size() == 0) {
            for (Map.Entry leftToRightEntry : leftToRightSamples.entrySet()) {
                int leftIdx = (Integer) leftToRightEntry.getKey();
                leftToRightNegSamples.put(leftIdx, new HashSet<>());
                Set<Integer> rightSet = trainLeftToRightSet.get(leftIdx);
                Set<Integer> rightNegSamples = leftToRightNegSamples.get(leftIdx);
                for (Integer posIdx : (Set<Integer>)leftToRightEntry.getValue()) {
                    int negIdx;
                    do {
                        negIdx = Randoms.uniform(numRight);
                    } while (rightSet.contains(negIdx) || rightNegSamples.contains(negIdx));
                    rightNegSamples.add(negIdx);
                    rightNegUnion.add(negIdx);
                    leftPosRightNegRightTable.put(leftIdx, posIdx, negIdx);
                    leftNegRightPosRightTable.put(leftIdx, negIdx, posIdx);
                    if (!negRightToLeftSamples.containsKey(negIdx)) {
                        Set<Integer> leftSet = new HashSet<>();
                        leftSet.add(leftIdx);
                       negRightToLeftSamples.put(negIdx, leftSet);
                    } else {
                        Set<Integer> leftSet = negRightToLeftSamples.get(negIdx);
                        leftSet.add(leftIdx);
                    }
                }
            }
        }
    }

    public void leftNegSampling(int batchSize) throws LibrecException {
        if (rightToLeftNegSamples.size() == 0 && leftToRightSamples.size() == 0) {
            sampling(batchSize);
        }
        if (rightToLeftNegSamples.size() != 0) {
            rightToLeftNegSamples.clear();
            rightPosLeftNegLeftTable.clear();
            rightNegLeftPosLeftTable.clear();
        }
        if (rightToLeftNegSamples.size() == 0) {
            for (Map.Entry rightToLeftEntry : rightToLeftNegSamples.entrySet()) {
                int rightIdx = (Integer) rightToLeftEntry.getKey();
                rightToLeftNegSamples.put(rightIdx, new HashSet<>());
                Set<Integer> leftSet = trainRightToLeftSet.get(rightIdx);
                Set<Integer> leftNegSamples = rightToLeftNegSamples.get(rightIdx);
                for (Integer posIdx : (Set<Integer>) rightToLeftEntry.getValue()) {
                    int negIdx;
                    do {
                        negIdx = Randoms.uniform(numLeft);
                    } while (leftSet.contains(negIdx) || leftNegSamples.contains(negIdx));
                    leftNegSamples.add(negIdx);
                    leftNegUnion.add(negIdx);
                    rightPosLeftNegLeftTable.put(rightIdx, posIdx, negIdx);
                    rightNegLeftPosLeftTable.put(rightIdx, negIdx, posIdx);
                }
            }
        }
    }
    protected void setup() {
        if (trainLeftToRightSet == null) {
            trainLeftToRightSet = getRowColumnsSet(trainSet, numLeft);
        }
        if (trainRightToLeftSet == null) {
            trainRightToLeftSet = getColumnRowsSet(trainSet, numRight);
        }
        leftToRightSamples = new HashMap<>(numLeft * 4 / 3 + 1);
        leftToRightNegSamples = new HashMap<>(numLeft * 4 / 3 + 1);
        negRightToLeftSamples = new HashMap<>(numRight * 4 / 3 + 1);
        rightToLeftSamples = new HashMap<>(numRight * 4 / 3 + 1);
        rightToLeftNegSamples = new HashMap<>(numRight * 4 / 3 + 1);
        leftUnion = new HashSet<>(numLeft);
        leftNegUnion = new HashSet<>(numLeft);
        rightUnion = new HashSet<>(numRight);
        rightNegUnion = new HashSet<>(numRight);
        leftPosRightNegRightTable = HashBasedTable.create();
        leftNegRightPosRightTable = HashBasedTable.create();
        rightPosLeftNegLeftTable = HashBasedTable.create();
        rightNegLeftPosLeftTable = HashBasedTable.create();
    }


    protected List<Set<Integer>> getRowColumnsSet(SequentialAccessSparseMatrix sparseMatrix, int numRows) {
        List<Set<Integer>> tempRowColumnsSet = new ArrayList<>();
        for (int rowIdx = 0; rowIdx < numRows; ++rowIdx) {
            int[] columnIndices = sparseMatrix.row(rowIdx).getIndices();
            Integer[] inputBoxed = org.apache.commons.lang.ArrayUtils.toObject(columnIndices);
            List<Integer> columnList = Arrays.asList(inputBoxed);
            tempRowColumnsSet.add(new HashSet<>(columnList));
        }
        return tempRowColumnsSet;
    }

    protected List<Set<Integer>> getColumnRowsSet(SequentialAccessSparseMatrix sparseMatrix, int numCols) {
        List<Set<Integer>> tempColumnRowsSet = new ArrayList<>();
        for (int colIdx = 0; colIdx < numCols; ++colIdx) {
            int[] rowIndices = sparseMatrix.column(colIdx).getIndices();
            Integer[] inputBoxed = org.apache.commons.lang.ArrayUtils.toObject(rowIndices);
            List<Integer> rowList = Arrays.asList(inputBoxed);
            tempColumnRowsSet.add(new HashSet<>(rowList));
        }
        return tempColumnRowsSet;
    }

    public Map<Integer, Set<Integer>> getLeftToRightSamples() {
        return leftToRightSamples;
    }

    public Map<Integer, Set<Integer>> getRightToLeftSamples() {
        return rightToLeftSamples;
    }

    public Map<Integer, Set<Integer>> getLeftToRightNegSamples() {
        return leftToRightNegSamples;
    }

    public Map<Integer, Set<Integer>> getNegRightToLeftSamples() {
        return negRightToLeftSamples;
    }

    public Map<Integer, Set<Integer>> getRightToLeftNegSamples() {
        return rightToLeftNegSamples;
    }

    public Set<Integer> leftLogicalOR(Set<Integer> set) {
        Set<Integer> union = new HashSet<>(leftUnion);
        union.addAll(set);
        return union;
    }

    public Set<Integer> rightLogicalOR(Set<Integer> set) {
        Set<Integer> union = new HashSet<>(rightUnion);
        union.addAll(set);
        return union;
    }

    public Set<Integer> getLeftNegUnion() {
        return leftNegUnion;
    }

    public Set<Integer> getLeftUnion() {
        return leftUnion;
    }

    public Set<Integer> getRightUnion() {
        return rightUnion;
    }

    public Set<Integer> getRightNegUnion() {
        return rightNegUnion;
    }

    public Table<Integer, Integer, Integer> getLeftNegRightPosRightTable() {
        return leftNegRightPosRightTable;
    }

    public Table<Integer, Integer, Integer> getLeftPosRightNegRightTable() {
        return leftPosRightNegRightTable;
    }

    public Table<Integer, Integer, Integer> getRightNegLeftPosLeftTable() {
        return rightNegLeftPosLeftTable;
    }

    public Table<Integer, Integer, Integer> getRightPosLeftNegLeftTable() {
        return rightPosLeftNegLeftTable;
    }
}
