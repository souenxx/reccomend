package batch;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import net.librec.common.LibrecException;
import net.librec.conf.Configuration;
import net.librec.math.algorithm.Randoms;
import net.librec.math.structure.SequentialAccessSparseMatrix;

import java.util.*;

public class PosNegBatchSet extends BatchSet{
    protected Table<Integer, Integer, Integer> leftPosToNegTables;
    protected Table<Integer, Integer, Integer> leftNegToPosTables;
    protected Table<Integer, Integer, Integer> posRightToNegTables;
    protected Table<Integer, Integer, Integer> negRightToPosTables;

    protected Map<Integer, Set<Integer>> leftToNegRightSamples;
    protected Map<Integer, Set<Integer>> negRightToLeftSamples;
    protected Map<Integer, Set<Integer>> rightToNegLeftSamples;
    protected Map<Integer, Set<Integer>> negLeftToRightSamples;
    public PosNegBatchSet(Configuration conf, SequentialAccessSparseMatrix trainMatrix) {
        super(conf, trainMatrix);
        leftPosToNegTables = HashBasedTable.create();
        posRightToNegTables = HashBasedTable.create();
        leftNegToPosTables = HashBasedTable.create();
        negRightToPosTables = HashBasedTable.create();
        leftToNegRightSamples = new HashMap<>();
        negLeftToRightSamples = new HashMap<>();
        negRightToLeftSamples = new HashMap<>();
        rightToNegLeftSamples = new HashMap<>();


    }

    @Override
    public void sampling(int batchSize) throws LibrecException {
        super.sampling(batchSize);
        String samplingType = conf.get("batch.negSampling.type", "right");
        if (samplingType.equals("right")) {
            if (leftToRightSamples.size() == 0) {
                sampling(batchSize);
            }
            if (leftToNegRightSamples.size() != 0 || leftPosToNegTables.size() != 0) {
                leftToNegRightSamples.clear();
                leftPosToNegTables.clear();
                leftNegToPosTables.clear();
            }
            if (leftToNegRightSamples.size() == 0  && leftPosToNegTables.size() == 0) {
                negSampling(leftToRightSamples, leftToRightNegSamples, negRightToLeftSamples,leftPosToNegTables, leftNegToPosTables, trainLeftToRightSet, numRight, rightNegUnion);
                for (Map.Entry entry : leftToRightSamples.entrySet()) {
                    int leftIdx = (Integer) entry.getKey();
                    leftToRightNegSamples.put(leftIdx, new HashSet<>());
                    Set<Integer> posSet = trainLeftToRightSet.get(leftIdx);
                    Set<Integer> negSamples = leftToRightNegSamples.get(leftIdx);
                    for (Integer posIdx : posSet) {
                        int negIdx;
                        do {
                            negIdx = Randoms.uniform(numRight);
                        } while (posSet.contains(negIdx) || negSamples.contains(negIdx));
                        negSamples.add(negIdx);
                        leftPosToNegTables.put(leftIdx, posIdx, negIdx);
                        leftNegToPosTables.put(leftIdx, negIdx, posIdx);
                    }
                }
            }
        } else if (samplingType.equals("left")) {
            if (rightToLeftSamples.size() == 0) {
                sampling(batchSize);
            }
            if (rightToNegLeftSamples.size() != 0 || posRightToNegTables.size() != 0) {
                rightToNegLeftSamples.clear();
                posRightToNegTables.clear();
                negRightToPosTables.clear();
            }
            if (rightToNegLeftSamples.size() == 0 && posRightToNegTables.size() == 0) {
                negSampling(rightToLeftSamples, rightToLeftNegSamples, negLeftToRightSamples,posRightToNegTables, negRightToPosTables, trainRightToLeftSet, numLeft, leftNegUnion);
            }
        }

    }

    protected void negSampling(Map<Integer, Set<Integer>> neuToPosSamples, Map<Integer, Set<Integer>> neuToNegSamples, Map<Integer, Set<Integer>> negToNeuSamples, Table<Integer, Integer, Integer> posToNegTables,
                             Table<Integer, Integer, Integer> negToPosTables, List<Set<Integer>> trainSet, int num, Set<Integer> negUnion) {
       for (Map.Entry en : neuToPosSamples.entrySet()) {
           int neuIdx = (Integer) en.getKey();
           neuToNegSamples.put(neuIdx, new HashSet<>());
           Set<Integer> posSet = trainSet.get(neuIdx);
           Set<Integer> negSamples = neuToNegSamples.get(neuIdx);
           for (Integer posIdx : posSet) {
               int negIdx;
               do {
                   negIdx = Randoms.uniform(num);
               } while (posSet.contains(negIdx) || negSamples.contains(negIdx));
               negSamples.add(negIdx);
               posToNegTables.put(neuIdx, posIdx, negIdx);
               negToPosTables.put(neuIdx, negIdx, posIdx);
               negUnion.add(negIdx);
               if (!negToNeuSamples.containsKey(negIdx)) {
                   negToNeuSamples.put(negIdx, new HashSet<>(neuIdx));
               } else {
                   Set<Integer> neuSamples = negToNeuSamples.get(negIdx);
                   neuSamples.add(neuIdx);
               }
           }
       }
    }

    public Map<Integer, Set<Integer>> getLeftToRightNegSamples() {
        return leftToNegRightSamples;
    }

    public Map<Integer, Set<Integer>> getNegRightToLeftSamples() {
        return negRightToLeftSamples;
    }

    public Map<Integer, Set<Integer>> getRightToNegLeftSamples() {
        return rightToNegLeftSamples;
    }

    public Map<Integer, Set<Integer>> getNegLeftToRightSamples() {
        return negLeftToRightSamples;
    }

    public Integer getNegRight(int neuIdx, int posIdx) {
        return leftPosToNegTables.get(neuIdx, posIdx);
    }

    public Integer getPosRight(int neuIdx, int negIdx) {
        return leftNegToPosTables.get(neuIdx, negIdx);
    }

    public Integer getNegLeft(int neuIdx, int posIdx) {
        return posRightToNegTables.get(neuIdx, posIdx);
    }

    public Integer getPosLeft(int neuIdx, int negIdx) {
        return negRightToPosTables.get(neuIdx, negIdx);
    }

}
