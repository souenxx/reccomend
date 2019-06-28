package recommend;

import com.google.common.collect.HashBiMap;
import it.unimi.dsi.fastutil.ints.IntOpenHashSet;
import net.librec.common.LibrecException;
import net.librec.math.algorithm.Maths;
import net.librec.math.algorithm.Randoms;
import net.librec.math.structure.*;
import net.librec.util.ArrayUtils;
import org.omg.CORBA.INTERNAL;

import java.lang.reflect.Array;
import java.util.*;
import java.util.Vector;

public class CMFSGDRecommender extends CMFRecommender{
    protected double epsilon;
    protected double eta;
    protected int batchSize;

    @Override
    protected void setup() throws LibrecException {
        super.setup();
        //userFactors.init(1);
        //itemFactors.init(1);
        //sideFactors.init(1);

    }
    @Override
    protected void trainModel() throws LibrecException{
        double tradeOff = conf.getDouble("rec.tradeoff", 1.0);
        batchSize = conf.getInt("rec.sgd.batchSize", 100);
        epsilon = conf.getDouble("rec.sgd.epsilon", 1e-8);
        eta = conf.getDouble("rec.sgd.eta", 1.0);
        double[][] userFactorsLearnRate = new double[numFactors][numUsers];
        double[][] itemFactorsLearnRate = new double[numFactors][numItems];
        double[][] sideFactorsLearnRate = new double[numFactors][numberOfSides];

        List<Set<Integer>> userItemsSet = getRowColumnsSet(trainMatrix, numUsers);
        List<Set<Integer>> userInfosSet = getRowColumnsSet(sideRatingMatrix, numUsers);

        for (int iter = 1; iter <= numIterations; iter++) {

            if (commonKey.equals("user")) {
                loss = 0.0d;
                int maxSampleSize = (trainMatrix.getNumEntries() + sideRatingMatrix.getNumEntries()) / (2 * batchSize);
                for (int sampleCount = 0; sampleCount < maxSampleSize; sampleCount++) {
                    //update itemFactors
                    //randomly draw batch size (userIdx, itemIdx)
                    int userItemSample = 0;
                    Map<Integer, Set<Integer>> batchUserItemsSet = new HashMap<>(numUsers * 4/3 + 1);
                    Map<Integer, Set<Integer>> batchItemUsersSet = new HashMap<>(numItems * 4/3 + 1);
                    Set<Integer> batchUsersUnion = new HashSet<>(numUsers);

                    while (userItemSample < batchSize) {
                        int userIdx = Randoms.uniform(numUsers);
                        Set<Integer> itemSet = userItemsSet.get(userIdx);
                        if (itemSet.size() == 0 || itemSet.size() == numItems)
                            continue;
                        int[] itemIndices = trainMatrix.row(userIdx).getIndices();
                        int itemIdx = itemIndices[Randoms.uniform(itemIndices.length)];

                        if (!batchUserItemsSet.containsKey(userIdx)) {
                            Set<Integer> batchItemsSet = new HashSet<>();
                            batchItemsSet.add(itemIdx);
                            batchUserItemsSet.put(userIdx, batchItemsSet);
                        } else if (!batchUserItemsSet.get(userIdx).contains(itemIdx)) {
                            Set<Integer> batchItemsSet = batchUserItemsSet.get(userIdx);
                            batchItemsSet.add(itemIdx);
                        } else {
                            continue;
                        }

                        if (!batchItemUsersSet.containsKey(itemIdx)) {
                            Set<Integer> batchUsersSet = new HashSet<>();
                            batchUsersSet.add(userIdx);
                            batchItemUsersSet.put(itemIdx, batchUsersSet);
                        } else {
                            Set<Integer> batchUsersSet = batchItemUsersSet.get(itemIdx);
                            batchUsersSet.add(userIdx);
                        }
                        batchUsersUnion.add(userIdx);
                        userItemSample++;
                    }

                    //randomly draw batch size (userIdx, infoIdx)
                    int userSideInfoSample = 0;

                    Map<Integer, Set<Integer>> batchUserInfosSet = new HashMap<>();
                    Map<Integer, Set<Integer>> batchInfoUsersSet = new HashMap<>();
                    while (userSideInfoSample < batchSize) {
                        int userSideIdx = Randoms.uniform(numUsers);
                        Set<Integer> infoSet = userInfosSet.get(userSideIdx);
                        if (infoSet.size() == 0 || infoSet.size() == numberOfSides)
                            continue;
                        int[] infoIndices = sideRatingMatrix.row(userSideIdx).getIndices();
                        int infoIdx = infoIndices[Randoms.uniform(infoIndices.length)];

                        if (!batchUserInfosSet.containsKey(userSideIdx)) {
                            Set<Integer> batchInfosSet = new HashSet<>();
                            batchInfosSet.add(infoIdx);
                            batchUserInfosSet.put(userSideIdx, batchInfosSet);
                        } else if (!batchUserInfosSet.get(userSideIdx).contains(infoIdx)) {
                            Set<Integer> batchInfosSet = batchUserInfosSet.get(userSideIdx);
                            batchInfosSet.add(infoIdx);
                        } else {
                            continue;
                        }

                        if (!batchInfoUsersSet.containsKey(infoIdx)) {
                            Set<Integer> batchUsersSet = new HashSet<>();
                            batchUsersSet.add(userSideIdx);
                            batchInfoUsersSet.put(infoIdx, batchUsersSet);
                        } else {
                            Set<Integer> batchUsersSet = batchInfoUsersSet.get(infoIdx);
                            batchUsersSet.add(userSideIdx);
                        }
                        batchUsersUnion.add(userSideIdx);
                        userSideInfoSample++;
                    }

                    //update itemFactors by mini batch sgd
                    for (Map.Entry itemUsers : batchItemUsersSet.entrySet()) {
                        int itemIdx = (Integer) itemUsers.getKey();
                        Set<Integer> batchUsersSet = batchItemUsersSet.get(itemIdx);
                        VectorBasedDenseVector userRatingsVector = new VectorBasedDenseVector(batchUsersSet.size());
                        VectorBasedDenseVector userPredictsVector = new VectorBasedDenseVector(batchUsersSet.size());
                        DenseMatrix batchUserFactors = new DenseMatrix(batchUsersSet.size(), numFactors);

                        //for (Integer userIdx : batchUsersSet) {
                        int index = 0;
                        for (Integer userIdx : (Set<Integer>) itemUsers.getValue()) {
                            //userRatingsVector.set(userIdx, trainMatrix.get(userIdx, itemIdx));
                            //userPredictsVector.set(userIdx, predict(userIdx, itemIdx));
                            userRatingsVector.set(index, trainMatrix.get(userIdx, itemIdx));
                            userPredictsVector.set(index, predict(userIdx, itemIdx));

                            batchUserFactors.set(index, userFactors.column(userIdx));
                            double lossError = (trainMatrix.get(userIdx, itemIdx) - predict(userIdx, itemIdx));
                            loss += lossError * lossError;
                            index++;
                        }

                        for (int factorIdx = 0; factorIdx < numFactors; factorIdx++) {
                            //VectorBasedDenseVector factorUsersVector = (VectorBasedDenseVector) userFactors.row(factorIdx);
                            MatrixBasedDenseVector factorUsersVector = (MatrixBasedDenseVector) batchUserFactors.column(factorIdx);
                            double realRatingValue = factorUsersVector.dot(userRatingsVector);
                            double estmRatingValue = factorUsersVector.dot(userPredictsVector);
                            double error = tradeOff * (realRatingValue - estmRatingValue);
                            //Adagrad
                            itemFactorsLearnRate[factorIdx][itemIdx] += error * error;
                            double del = adagrad(itemFactorsLearnRate[factorIdx][itemIdx], error, batchUsersSet.size());
                            itemFactors.plus(factorIdx, itemIdx, del);
                            if (itemFactors.get(factorIdx, itemIdx) < 0) {
                                itemFactors.set(factorIdx, itemIdx, 0.0);
                            }
                        }
                    }



                    //for (int sideIdx = 0; sideIdx < numberOfSides; sideIdx++) {
                    for (Map.Entry infoUsers : batchInfoUsersSet.entrySet()) {
                        int sideIdx = (Integer) infoUsers.getKey();
                        int infoUsersSize = ((Set<Integer>) infoUsers.getValue()).size();
                        VectorBasedDenseVector userSideRatingsVector = new VectorBasedDenseVector(infoUsersSize);
                        VectorBasedDenseVector userSidePredictsVector = new VectorBasedDenseVector(infoUsersSize);
                        DenseMatrix batchUserFactors = new DenseMatrix(infoUsersSize, numFactors);

                        //for (Integer userIdx : batchUsersSet) {
                        int index = 0;
                        for (Integer userIdx : (Set<Integer>) infoUsers.getValue()) {
                            //userSideRatingsVector.set(userIdx, sideRatingMatrix.get(userIdx, sideIdx));
                            //userSidePredictsVector.set(userIdx, predSideRating(userIdx, sideIdx));
                            userSideRatingsVector.set(index, sideRatingMatrix.get(userIdx, sideIdx));
                            userSidePredictsVector.set(index, predSideRating(userIdx, sideIdx));
                            batchUserFactors.set(index, userFactors.column(userIdx));
                            double lossError = (sideRatingMatrix.get(userIdx, sideIdx) - predSideRating(userIdx, sideIdx));
                            loss += lossError * lossError;
                            index++;
                        }

                        for (int factorIdx = 0; factorIdx < numFactors; factorIdx++) {
                            //VectorBasedDenseVector factorSideUsersVector = (VectorBasedDenseVector) userFactors.row(factorIdx);
                            MatrixBasedDenseVector factorSideUsersVector = (MatrixBasedDenseVector) batchUserFactors.column(factorIdx);
                            double realSideRatingValue = factorSideUsersVector.dot(userSideRatingsVector);
                            double estmSideRatingValue = factorSideUsersVector.dot(userSidePredictsVector);
                            double error = (1.0 - tradeOff) * (realSideRatingValue - estmSideRatingValue);
                            sideFactorsLearnRate[factorIdx][sideIdx] += error * error;
                            double del = adagrad(sideFactorsLearnRate[factorIdx][sideIdx], error, ((Set<Integer>) infoUsers.getValue()).size());
                            //sideFactors.plus(factorIdx, sideIdx,
                            //        (eta / (Math.sqrt(sideFactorsLearnRate[factorIdx][sideIdx]) + epsilon)) * error / (double) batchSize
                            //);
                            sideFactors.plus(factorIdx, sideIdx, del);
                            if (sideFactors.get(factorIdx, sideIdx) < 0) {
                                sideFactors.set(factorIdx, sideIdx, 0.0);
                            }

                        }

                    }

                    for (Integer userIdx : batchUsersUnion) {
                            Set<Integer> batchItemsSet = new HashSet<>();
                            Set<Integer> batchInfosSet = new HashSet<>();
                            VectorBasedDenseVector itemRatingsVector;
                            VectorBasedDenseVector itemPredictsVector;
                            VectorBasedDenseVector infoSideRatingsVector;
                            VectorBasedDenseVector infoSidePredictsVector;
                            DenseMatrix batchItemFactors, batchInfoFactors;
                            if (batchUserItemsSet.containsKey(userIdx)) {
                                batchItemsSet = batchUserItemsSet.get(userIdx);
                                itemRatingsVector = new VectorBasedDenseVector(batchItemsSet.size());
                                itemPredictsVector = new VectorBasedDenseVector(batchItemsSet.size());
                                batchItemFactors = new DenseMatrix(batchItemsSet.size(), numFactors);

                                int index = 0;
                                for (Integer itemIdx : batchItemsSet) {
                                    //itemRatingsVector.set(itemIdx, trainMatrix.get(userIdx, itemIdx));
                                    //itemPredictsVector.set(itemIdx, predict(userIdx, itemIdx));
                                    itemRatingsVector.set(index, trainMatrix.get(userIdx, itemIdx));
                                    itemPredictsVector.set(index, predict(userIdx, itemIdx));
                                    batchItemFactors.set(index, itemFactors.column(itemIdx));
                                    double lossError = trainMatrix.get(userIdx, itemIdx) - predict(userIdx, itemIdx);
                                    loss += lossError * lossError;
                                    index++;
                                }
                            } else {
                                itemRatingsVector = new VectorBasedDenseVector(0);
                                itemPredictsVector = new VectorBasedDenseVector(0);
                                batchItemFactors = new DenseMatrix(0,0);
                            }
                            if (batchUserInfosSet.containsKey(userIdx)) {
                                batchInfosSet = batchUserInfosSet.get(userIdx);
                                infoSideRatingsVector = new VectorBasedDenseVector(batchInfosSet.size());
                                infoSidePredictsVector = new VectorBasedDenseVector(batchInfosSet.size());
                                batchInfoFactors = new DenseMatrix(batchInfosSet.size(), numFactors);

                                int index = 0;
                                for (Integer infoIdx : batchInfosSet) {
                                    //infoSideRatingsVector.set(infoIdx, sideRatingMatrix.get(userIdx, infoIdx));
                                    //infoSidePredictsVector.set(infoIdx, predSideRating(userIdx, infoIdx));
                                    infoSideRatingsVector.set(index, sideRatingMatrix.get(userIdx, infoIdx));
                                    infoSidePredictsVector.set(index, predSideRating(userIdx, infoIdx));
                                    batchInfoFactors.set(index, sideFactors.column(infoIdx));
                                    double lossError = sideRatingMatrix.get(userIdx, infoIdx) - predSideRating(userIdx, infoIdx);
                                    loss += lossError * lossError;
                                    index++;
                                }
                            } else {
                                infoSideRatingsVector = new VectorBasedDenseVector(0);
                                infoSidePredictsVector = new VectorBasedDenseVector(0);
                                batchInfoFactors = new DenseMatrix(0,0);
                            }

                            for (int factorIdx = 0; factorIdx < numFactors; factorIdx++) {
                                //VectorBasedDenseVector factorItemsVector = (VectorBasedDenseVector) itemFactors.row(factorIdx);
                                MatrixBasedDenseVector factorItemsVector = (MatrixBasedDenseVector) batchItemFactors.column(factorIdx);
                                double realRatingValue = factorItemsVector.dot(itemRatingsVector);
                                double estmRatingValue = factorItemsVector.dot(itemPredictsVector);
                                double realError = tradeOff * (realRatingValue - estmRatingValue);

                                //VectorBasedDenseVector factorInfosVector = (VectorBasedDenseVector) sideFactors.row(factorIdx);
                                MatrixBasedDenseVector factorInfosVector = (MatrixBasedDenseVector) batchInfoFactors.column(factorIdx);
                                double realSideRatingValue = factorInfosVector.dot(infoSideRatingsVector);
                                double estmSideRatingValue = factorInfosVector.dot(infoSidePredictsVector);
                                double sideError = (1.0 - tradeOff) * (realSideRatingValue - estmSideRatingValue);
                                double error = realError + sideError;
                                userFactorsLearnRate[factorIdx][userIdx] += error * error;
                                double del = adagrad(userFactorsLearnRate[factorIdx][userIdx], error, batchItemsSet.size() + batchInfosSet.size());
                                userFactors.plus(factorIdx, userIdx, del);
                                if (userFactors.get(factorIdx, userIdx) < 0) {
                                    userFactors.set(factorIdx, userIdx, 0.0);
                                }
                            }

                    }
                }

                LOG.info("iter:" + iter + ", loss:" + loss);
            }
        }
    }

    protected double adagrad(double sumSquareGrad, double grad, int subBatchSize) {
        return (1.0 / (double) subBatchSize) * (eta / (Math.sqrt(sumSquareGrad) + epsilon)) * grad;
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

}
