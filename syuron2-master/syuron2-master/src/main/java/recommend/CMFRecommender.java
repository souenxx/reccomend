package recommend;

import com.google.common.collect.BiMap;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.HashBiMap;
import com.google.common.collect.Table;
import net.librec.common.LibrecException;
import net.librec.math.structure.*;
import net.librec.recommender.TensorRecommender;
import org.apache.commons.lang.StringUtils;

import javax.xml.crypto.Data;
import java.util.HashMap;
import java.util.Map;

public class CMFRecommender extends TensorRecommender{
    protected int numberOfSides;
    protected int numberOfCommons;
    protected int numberOfRatingInfo;
    protected DenseMatrix userFactors;
    protected DenseMatrix itemFactors;
    protected DenseMatrix commonFactors;
    protected DenseMatrix infoFactors;
    protected DenseMatrix sideFactors;
    protected SequentialAccessSparseMatrix sideRatingMatrix;

    protected double lambdaU;
    protected double lambdaV;
    protected double lambdaW;

    protected SequentialAccessSparseMatrix trainMatrix;
    public BiMap<Integer, String> sideRatingPairsMappingData;

    protected String commonKey;

    @Override
    protected void setup() throws LibrecException {
        super.setup();
        lambdaU = conf.getDouble("rec.regularization.lambdau", 0.001);
        lambdaV = conf.getDouble("rec.regularization.lambdav", 0.001);
        lambdaW = conf.getDouble("rec.regularization.lambdaw", 0.001);

        commonKey = conf.get("rec.cmf.commonkey");

        BiMap<Integer, String> userPairsMappingData = DataFrame.getInnerMapping("user").inverse();
        BiMap<Integer, String> itemPairsMappingData = DataFrame.getInnerMapping("item").inverse();
        sideRatingPairsMappingData = DataFrame.getInnerMapping("side").inverse();
        BiMap<String, Integer> sideDict = HashBiMap.create();

        trainMatrix = trainTensor.rateMatrix();

        int commonEntryKey;
        int sideEntryKey;
        BiMap<String, Integer> encodePairsMappingData = HashBiMap.create();
        if (commonKey.equals("user")) {
            encodePairsMappingData = DataFrame.getInnerMapping("user");
            numberOfCommons = numUsers;
            numberOfRatingInfo = numItems;
            commonEntryKey = 0;
            sideEntryKey = 1;
        } else if (commonKey.equals("item")) {
            encodePairsMappingData = DataFrame.getInnerMapping("item");
            numberOfCommons = numItems;
            numberOfRatingInfo = numUsers;
            commonEntryKey = 1;
            sideEntryKey = 0;
        } else {
            throw new LibrecException("common key is not user or item");
        }

        numberOfSides = 0;
        Table<Integer, Integer, Double> sideRatingTable = HashBasedTable.create();
        for (TensorEntry te: trainTensor) {
            int[] entryKeys = te.keys();
            int userIndex = entryKeys[0];
            int itemIndex = entryKeys[1];
            int sideRatingPairsIndex = entryKeys[2];

            String sideRatingPairsString = sideRatingPairsMappingData.get(sideRatingPairsIndex);
            if (sideRatingPairsString.isEmpty())
                continue;
            String[] sRPList = sideRatingPairsString.split(" ");


            for (String srp : sRPList) {
                String commonIdxString = srp.split(";")[commonEntryKey];
                int commonEncodeIdx = Integer.valueOf(encodePairsMappingData.get(commonIdxString));
                String sideIdxString = srp.split(";")[sideEntryKey];
                double sideRating = Double.valueOf(srp.split(";")[2]);
                if (!sideDict.containsKey(sideIdxString) && !StringUtils.isEmpty(sideIdxString)) {
                    sideDict.put(sideIdxString, numberOfSides);
                    numberOfSides++;
                }

                if (commonKey.equals("user"))
                    sideRatingTable.put(commonEncodeIdx, sideDict.get(sideIdxString), sideRating);
                else
                    sideRatingTable.put(sideDict.get(sideIdxString), commonEncodeIdx, sideRating);
            }
        }

        if (commonKey.equals("user"))
            sideRatingMatrix = new SequentialAccessSparseMatrix(numUsers, numberOfSides, sideRatingTable);
        else
            sideRatingMatrix = new SequentialAccessSparseMatrix(numItems, numberOfSides, sideRatingTable);

        //create U, V, W
        userFactors = new DenseMatrix(numFactors, numUsers);
        userFactors.init(0.01);
        itemFactors = new DenseMatrix(numFactors, numItems);
        itemFactors.init(0.01);
        infoFactors = new DenseMatrix(numFactors, numberOfRatingInfo);
        infoFactors.init(0.01);

        sideFactors = new DenseMatrix(numFactors, numberOfSides);
        sideFactors.init(0.01);

        LOG.info("numUsers:" + numUsers);
        LOG.info("numItems:" + numItems);
        LOG.info("numSides:" + numberOfSides);
    }

    @Override
    protected void trainModel() throws LibrecException{
        double tradeOff = conf.getDouble("rec.tradeoff", 1.0);
        for (int iter = 1; iter <= numIterations; iter++) {
            loss = 0.0d;

            if (commonKey.equals("user")) {
                //update userFactors
                for (int userIdx = 0; userIdx < numUsers; userIdx++) {
                    SequentialSparseVector itemRatingsVector = trainMatrix.row(userIdx);
                    SequentialSparseVector infoSideRatingsVector = sideRatingMatrix.row(userIdx);

                    if (itemRatingsVector.getNumEntries() > 0 && infoSideRatingsVector.getNumEntries() > 0) {
                        VectorBasedDenseVector itemPredictsVector = new VectorBasedDenseVector(numItems);
                        VectorBasedDenseVector sidePredictsVector = new VectorBasedDenseVector(numberOfSides);

                        for (int i = 0; i < itemRatingsVector.getNumEntries(); i++) {
                            int itemIdx = itemRatingsVector.getIndexAtPosition(i);
                            itemPredictsVector.set(itemIdx, predict(userIdx, itemIdx));
                        }

                        for (int i = 0; i < infoSideRatingsVector.getNumEntries(); i++) {
                            int sideIdx = infoSideRatingsVector.getIndexAtPosition(i);
                            sidePredictsVector.set(sideIdx, predSideRating(userIdx, sideIdx));
                        }

                        for (int factorIdx = 0; factorIdx < numFactors; factorIdx++) {
                            VectorBasedDenseVector factorItemsVector = (VectorBasedDenseVector) itemFactors.row(factorIdx);
                            VectorBasedDenseVector factorSidesVector = (VectorBasedDenseVector) sideFactors.row(factorIdx);
                            double realRatingValue = factorItemsVector.dot(itemRatingsVector);
                            double estmRatingValue = factorItemsVector.dot(itemPredictsVector) + 1e-9;

                            double realSideRatingValue = factorSidesVector.dot(infoSideRatingsVector);
                            double estmSideRatingValue = factorSidesVector.dot(sidePredictsVector);

                            userFactors.set(factorIdx, userIdx, userFactors.get(factorIdx, userIdx)
                                    * (((tradeOff) * realRatingValue + (1.0 - tradeOff) * realSideRatingValue) /
                                    ((tradeOff) * estmRatingValue + (1.0 - tradeOff) * estmSideRatingValue))
                            );
                        }
                    }
                }

                //update sideFactors
                if (tradeOff < 1.0) {
                    for (int sideIdx = 0; sideIdx < numberOfSides; sideIdx++) {
                        SequentialSparseVector userSideRatingsVector = sideRatingMatrix.column(sideIdx);
                        if (userSideRatingsVector.getNumEntries() > 0) {
                            VectorBasedDenseVector userSidePredictsVector = new VectorBasedDenseVector(numUsers);

                            for (int i = 0; i < userSideRatingsVector.getNumEntries(); i++) {
                                int userIdx = userSideRatingsVector.getIndexAtPosition(i);
                                userSidePredictsVector.set(userIdx, predSideRating(userIdx, sideIdx));
                            }

                            for (int factorIdx = 0; factorIdx < numFactors; factorIdx++) {
                                VectorBasedDenseVector factorUsersVector = (VectorBasedDenseVector) userFactors.row(factorIdx);

                                double realSideRatingValue = factorUsersVector.dot(userSideRatingsVector);
                                double estmSideRatingValue = factorUsersVector.dot(userSidePredictsVector) + 1e-9;

                                sideFactors.set(factorIdx, sideIdx, sideFactors.get(factorIdx, sideIdx)
                                        * (((1.0 - tradeOff) * realSideRatingValue) / ((1.0 - tradeOff) * estmSideRatingValue))
                                );
                            }
                        }
                    }
                }

                if (tradeOff > 0.0) {
                    for (int itemIdx = 0; itemIdx < numItems; itemIdx++) {
                        SequentialSparseVector userRatingsVector = trainMatrix.column(itemIdx);
                        if (userRatingsVector.getNumEntries() > 0) {
                            VectorBasedDenseVector userPredictsVector = new VectorBasedDenseVector(numUsers);

                            for (int i = 0; i < userRatingsVector.getNumEntries(); i++) {
                                int userIdx = userRatingsVector.getIndexAtPosition(i);
                                userPredictsVector.set(userIdx, predict(userIdx, itemIdx));
                            }

                            for (int factorIdx = 0; factorIdx < numFactors; factorIdx++) {
                                VectorBasedDenseVector factorUsersVector = (VectorBasedDenseVector) userFactors.row(factorIdx);
                                double realValue = factorUsersVector.dot(userRatingsVector);
                                double estmValue = factorUsersVector.dot(userPredictsVector) + 1e-9;

                                itemFactors.set(factorIdx, itemIdx, itemFactors.get(factorIdx, itemIdx)
                                        * ((tradeOff * realValue) / (tradeOff * estmValue))
                                );
                            }
                        }
                    }
                }

                loss = 0.0d;
                for (MatrixEntry matrixEntry : trainMatrix) {
                    int userIdx = matrixEntry.row();
                    int itemIdx = matrixEntry.column();
                    double rating = matrixEntry.get();

                    if (rating > 0) {
                        double ratingError = predict(userIdx, itemIdx) - rating;

                        loss += ratingError * ratingError;
                    }
                }

                for (MatrixEntry sideMatrixEntry : sideRatingMatrix) {
                    int userIdx = sideMatrixEntry.row();
                    int infoIdx = sideMatrixEntry.column();
                    double sideRating = sideMatrixEntry.get();

                    if (sideRating > 0) {
                        double sideRatingError =  predSideRating(userIdx, infoIdx) - sideRating;

                        loss += sideRatingError * sideRatingError;
                    }
                }
            }
            LOG.info("iter:" + iter + ", loss:" + loss);

        }
    }

    @Override
    protected double predict(int [] indices) {return predict(indices[0], indices[1]);}

    protected double predict(int u, int j) {
        return userFactors.column(u).dot(itemFactors.column(j));
    }


    protected double getCommonFactorValue(int commonIdx, int factorIdx) {
        if (commonKey.equals("user"))
            return userFactors.get(commonIdx, factorIdx);
        else
            return itemFactors.get(commonIdx, factorIdx);
    }

    protected void updateCommonFactors(int commonIdx, int factorIdx, double updateValue) {
        if (commonKey.equals("user"))
            userFactors.plus(commonIdx, factorIdx, updateValue);
        else
            itemFactors.plus(commonIdx, factorIdx, updateValue);
    }


    protected double predSideRating(int commonIdx, int sideIdx) {
        if (commonKey.equals("user"))
            return userFactors.column(commonIdx).dot(sideFactors.column(sideIdx));
        else
            return itemFactors.row(commonIdx).dot(sideFactors.row(sideIdx));
    }


    protected double normalize(double rating) {return 0;}

    //protected void setTable()
}
