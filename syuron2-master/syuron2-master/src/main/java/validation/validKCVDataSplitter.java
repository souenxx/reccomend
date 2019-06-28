package validation;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import com.sun.tools.javac.util.Name;
import net.librec.common.LibrecException;
import net.librec.conf.Configuration;
import net.librec.data.DataConvertor;
import net.librec.data.splitter.KCVDataSplitter;
import net.librec.data.splitter.RatioDataSplitter;
import net.librec.math.algorithm.Randoms;
import net.librec.math.structure.MatrixEntry;
import net.librec.math.structure.SequentialAccessSparseMatrix;
import net.librec.math.structure.SparseTensor;
import net.librec.util.Lists;

import java.util.*;


public class validKCVDataSplitter extends RatioDataSplitter {
    protected SequentialAccessSparseMatrix validPreferenceMatrix;
    protected SequentialAccessSparseMatrix validAssignMatrix;
    protected LinkedList<SequentialAccessSparseMatrix> validAssignMatrixList;
    protected LinkedList<SequentialAccessSparseMatrix> cloneValidAssignMatrixList;
    protected int cvNumver;
    protected int cvIndex;
    protected SequentialAccessSparseMatrix totalTestMatrix;
    public validKCVDataSplitter() {
    }

    public validKCVDataSplitter(Configuration conf) {this.conf = conf;}

    public void splitData(int kFold) {
       if (kFold > 0) {
           validAssignMatrix = validPreferenceMatrix.clone();
           int numRates = validPreferenceMatrix.getNumEntries();
           int numFold = kFold > numRates ? numRates : kFold;

           List<Map.Entry<Integer, Double>> rdm = new ArrayList<>(numRates);
           double indvCount = (numRates + 0.0) / numFold;
           for (int index = 0; index < numRates; index++) {
               rdm.add(new AbstractMap.SimpleImmutableEntry<Integer, Double>((int) (index / indvCount) + 1, Randoms.uniform()));
           }

           int[] fold = new int[numRates];
           Lists.sortList(rdm, true);
           for (int index = 0; index < numRates; index++) {
               fold[index] = rdm.get(index).getKey();
           }

           int i = 0;
           for (MatrixEntry matrixEntry : validAssignMatrix) {
               validAssignMatrix.setAtColumnPosition(matrixEntry.row(), matrixEntry.columnPosition(), fold[i++]);
           }

       }
        if (null == validAssignMatrixList) {
            List<Table<Integer, Integer, Integer>> tableList = new ArrayList<>(kFold + 1);
            for (int i = 0; i < kFold + 1; i++) {
                tableList.add(HashBasedTable.create());
            }
            for (MatrixEntry me : validAssignMatrix) {
                if (me.get() != 0) {
                    tableList.get((int) me.get()).put(me.row(), me.column(), 1);
                }
            }

            this.validAssignMatrixList = new LinkedList<>();
            for (int i = 1; i < kFold + 1; i++) {
                this.validAssignMatrixList.add(new SequentialAccessSparseMatrix(validAssignMatrix.rowSize(), validAssignMatrix.columnSize(), tableList.get(i)));
            }
            if (cloneValidAssignMatrixList == null) {
                cloneValidAssignMatrixList = (LinkedList<SequentialAccessSparseMatrix>) validAssignMatrixList.clone();
            }
        }

    }

    @Override
    public boolean nextFold() {
        if (this.validAssignMatrixList == null) {
            validAssignMatrixList = new LinkedList<>();
            return true;
        } else if (conf.get("data.split.valid").equals("test")) {
            if (assignMatrixList == null) {
                assignMatrixList = new LinkedList<>();
                return true;
            } else {
                assignMatrixList = null;
                return false;
            }
        } else {
            if (validAssignMatrixList.size() > 0) {
                SequentialAccessSparseMatrix validAssign = validAssignMatrixList.poll();
                trainMatrix = validPreferenceMatrix.clone();
                for (MatrixEntry matrixEntry : validPreferenceMatrix) {
                    if (validAssign.get(matrixEntry.row(), matrixEntry.column()) == 1) {
                        trainMatrix.setAtColumnPosition(matrixEntry.row(), matrixEntry.columnPosition(), 0.0D);
                    }
                }
                if (conf.get("data.split.valid").equals("valid")) {
                    testMatrix = validPreferenceMatrix.clone();


                    for (MatrixEntry matrixEntry : validPreferenceMatrix) {
                        if (validAssign.get(matrixEntry.row(), matrixEntry.column()) != 1) {
                            testMatrix.setAtColumnPosition(matrixEntry.row(), matrixEntry.columnPosition(), 0.0D);
                        }
                    }

                } else if (conf.get("data.split.valid").equals("test")) {
                    //SequentialAccessSparseMatrix assign = assignMatrixList.poll();
                    //testMatrix = preferenceMatrix.clone();

                    //for (MatrixEntry matrixEntry : preferenceMatrix) {
                     //  if (assign.get(matrixEntry.row(), matrixEntry.column()) != 1) {
                      //     testMatrix.setAtColumnPosition(matrixEntry.row(), matrixEntry.columnPosition(), 0.0D);
                       //}
                    if (assignMatrixList == null) {
                        assignMatrixList = new LinkedList<>();
                        return true;
                    } else {
                        assignMatrixList = new LinkedList<>();
                        return false;
                    }
                }
                trainMatrix.reshape();
                testMatrix.reshape();
                return true;
            } else {
                validAssignMatrixList = (LinkedList<SequentialAccessSparseMatrix>) cloneValidAssignMatrixList.clone();
                return false;
            }
        }
    }
    @Override
    public void splitData() throws LibrecException {
        super.splitData();
        validPreferenceMatrix = trainMatrix.clone();
        totalTestMatrix = testMatrix.clone();
        this.cvNumver = conf.getInt("data.splitter.cv.number", 5);
        if (null == validAssignMatrix) {
            splitData(this.cvNumver);
        }
    }

    public List<SequentialAccessSparseMatrix> getAssignMatrixList() {
        return this.validAssignMatrixList;
    }

    public SequentialAccessSparseMatrix getValidPreferenceMatrix() {
        return validPreferenceMatrix;
    }

    public SequentialAccessSparseMatrix getTotalTestMatrix() {
        return totalTestMatrix;
    }


}
