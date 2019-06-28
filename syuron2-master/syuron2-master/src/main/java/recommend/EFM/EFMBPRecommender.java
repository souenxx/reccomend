package recommend.EFM;


import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.commons.lang.StringUtils;

import com.google.common.collect.BiMap;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.HashBiMap;
import com.google.common.collect.Table;

import batch.BatchSet;
import net.librec.common.LibrecException;
import net.librec.math.algorithm.Maths;
import net.librec.math.structure.DataFrame;
import net.librec.math.structure.DenseMatrix;
import net.librec.math.structure.DenseVector;
import net.librec.math.structure.MatrixBasedDenseVector;
import net.librec.math.structure.SequentialAccessSparseMatrix;
import net.librec.math.structure.TensorEntry;
import net.librec.math.structure.VectorBasedDenseVector;
import net.librec.recommender.TensorRecommender;

public class EFMBPRecommender extends TensorRecommender{
    protected int numberOfFeatures;
    protected int explicitFeatureNum;
    protected int hiddenFeatureNum;
    protected double scoreScale;
    protected DenseMatrix featureMatrix;
    protected DenseMatrix userFeatureMatrix;
    protected DenseMatrix userHiddenMatrix;
    protected DenseMatrix itemFeatureMatrix;
    protected DenseMatrix itemHiddenMatrix;
    protected SequentialAccessSparseMatrix userFeatureAttention;
    protected SequentialAccessSparseMatrix itemFeatureQuality;
    protected double lambdaX;
    protected double lambdaY;
    protected double lambdaU;
    protected double lambdaH;
    protected double lambdaV;
    protected double epsilon;
    protected double eta;
    protected int batchSize;
    protected BiMap<String, Integer> featureDict;

    protected SequentialAccessSparseMatrix trainMatrix;

    public BiMap<Integer, String> featureSentimemtPairsMappingData;
    protected ArrayList<Map<Integer, Double>> userFeatureValuesList;
    protected DenseMatrix userFeatureValuesMatrix;
    protected DenseMatrix recItemFeatureValuesMatrix;
    protected boolean[] userFeatureValuesFlag;
    protected boolean[] recItemFeatureValuesFlag;
    protected ArrayList<VectorBasedDenseVector> userFeatureValuesVectorList;
    protected ArrayList<VectorBasedDenseVector> recItemFeatureValuesVectorList;

    boolean doExplain;
    boolean doRanking;

    double maxUserFeature, minUserFeature;
    double rescaling;

    /*
     * (non-Javadoc)
     *
     * @see net.librec.recommender.AbstractRecommender#setup()
     */
    @Override
    protected void setup() throws LibrecException {
        super.setup();
        scoreScale = maxRate - minRate;
        explicitFeatureNum = conf.getInt("rec.factor.explicit", 5);
        hiddenFeatureNum = numFactors - explicitFeatureNum;
        lambdaX = conf.getDouble("rec.regularization.lambdax", 0.001);
        lambdaY = conf.getDouble("rec.regularization.lambday", 0.001);
        lambdaU = conf.getDouble("rec.regularization.lambdau", 0.001);
        lambdaH = conf.getDouble("rec.regularization.lambdah", 0.001);
        lambdaV = conf.getDouble("rec.regularization.lambdav", 0.001);

        featureSentimemtPairsMappingData = DataFrame.getInnerMapping("sentiment").inverse();
        trainMatrix = trainTensor.rateMatrix();

        featureDict = HashBiMap.create();
        Map<Integer, String> userFeatureDict = new HashMap<>();
        Map<Integer, String> itemFeatureDict = new HashMap<Integer, String>();

        numberOfFeatures = 0;

        //set up ndcg////////////////////////////////////////
        userFeatureValuesList= new ArrayList<>(numUsers);
        //ArrayList initialize
        for (int var = 0; var < numUsers; var++) {
            userFeatureValuesList.add(var, null);
        }
        //featureValuesFlag initialize
        userFeatureValuesFlag = new boolean[numUsers];
        recItemFeatureValuesFlag = new boolean[numItems];
        for (int var = 0; var < numUsers; var++) userFeatureValuesFlag[var] = false;
        for (int var = 0; var < numItems; var++) recItemFeatureValuesFlag[var] = false;
        //initialize featureValuesMatrix
        userFeatureValuesMatrix = new DenseMatrix(numUsers, numberOfFeatures);
        recItemFeatureValuesMatrix = new DenseMatrix(numItems, numberOfFeatures);
        userFeatureValuesVectorList= new ArrayList<>(numUsers);
        for (int var = 0; var < numUsers; var++) {
            userFeatureValuesVectorList.add(var, null);
        }
        recItemFeatureValuesVectorList= new ArrayList<>(numItems);
        for (int var = 0; var < numItems; var++) {
            recItemFeatureValuesVectorList.add(var, null);
        }
        /////////////////////////////////////////////////////


        for (TensorEntry te : trainTensor) {
            int[] entryKeys = te.keys();
            int userIndex = entryKeys[0];
            int itemIndex = entryKeys[1];
            int featureSentimentPairsIndex = entryKeys[2];
            String featureSentimentPairsString = featureSentimemtPairsMappingData.get(featureSentimentPairsIndex)
                    .replaceAll("</endperson[0-9]+>", "");
            String[] fSPList = featureSentimentPairsString.split(" ");

            for (String p : fSPList) {
                String k = p.split(":")[0];
                if (!featureDict.containsKey(k) && !StringUtils.isEmpty(k)) {
                    featureDict.put(k, numberOfFeatures);
                    numberOfFeatures++;
                }
                if (userFeatureDict.containsKey(userIndex)) {
                    userFeatureDict.put(userIndex, userFeatureDict.get(userIndex) + " " + p);
                } else {
                    userFeatureDict.put(userIndex, p);
                }
                if (itemFeatureDict.containsKey(itemIndex)) {
                    itemFeatureDict.put(itemIndex, itemFeatureDict.get(itemIndex) + " " + p);
                } else {
                    itemFeatureDict.put(itemIndex, p);
                }
            }
        }

        // Create V,U1,H1,U2,H2
        featureMatrix = new DenseMatrix(numberOfFeatures, explicitFeatureNum);
        featureMatrix.init(0.01);
        userFeatureMatrix = new DenseMatrix(numUsers, explicitFeatureNum); //userFactors.getSubMatrix(0, userFactors.numRows() - 1, 0, explicitFeatureNum - 1);
        userFeatureMatrix.init(0.01);
        userHiddenMatrix = new DenseMatrix(numUsers, numFactors - explicitFeatureNum); // userFactors.getSubMatrix(0, userFactors.numRows() - 1, explicitFeatureNum, userFactors.numColumns() - 1);
        userHiddenMatrix.init(0.01);
        itemFeatureMatrix = new DenseMatrix(numItems, explicitFeatureNum);// itemFactors.getSubMatrix(0, itemFactors.numRows() - 1, 0, explicitFeatureNum - 1);
        itemFeatureMatrix.init(0.01);
        itemHiddenMatrix = new DenseMatrix(numItems, numFactors - explicitFeatureNum);// itemFactors.getSubMatrix(0, itemFactors.numRows() - 1, explicitFeatureNum, itemFactors.numColumns() - 1);
        itemHiddenMatrix.init(0.01);

        // compute UserFeatureAttention
        Table<Integer, Integer, Double> userFeatureAttentionTable = HashBasedTable.create();
        for (int u : userFeatureDict.keySet()) {
            double[] featureValues = new double[numberOfFeatures];
            String[] fList = userFeatureDict.get(u).split(" ");
            for (String a : fList) {
                if (!StringUtils.isEmpty(a)) {
                    int fin = featureDict.get(a.split(":")[0]);
                    featureValues[fin] += 1;
                }
            }
            for (int i = 0; i < numberOfFeatures; i++) {
                if (featureValues[i] != 0.0) {
                    double v = 1 + (scoreScale - 1) * (2 / (1 + Math.exp(-featureValues[i])) - 1);
                    userFeatureAttentionTable.put(u, i, v);
                }
            }
        }
        userFeatureAttention = new SequentialAccessSparseMatrix(numUsers, numberOfFeatures, userFeatureAttentionTable);

        // Compute ItemFeatureQuality
        Table<Integer, Integer, Double> itemFeatureQualityTable = HashBasedTable.create();
        for (int p : itemFeatureDict.keySet()) {
            double[] featureValues = new double[numberOfFeatures];
            String[] fList = itemFeatureDict.get(p).split(" ");
            for (String a : fList) {
                if (!StringUtils.isEmpty(a)) {
                    int fin = featureDict.get(a.split(":")[0]);
                    featureValues[fin] += Double.parseDouble(a.split(":")[1]);
                }
            }
            for (int i = 0; i < numberOfFeatures; i++) {
                if (featureValues[i] != 0.0) {
                    double v = 1 + (scoreScale - 1) / (1 + Math.exp(-featureValues[i]));
                    itemFeatureQualityTable.put(p, i, v);
                }
            }
        }
        itemFeatureQuality = new SequentialAccessSparseMatrix(numItems, numberOfFeatures, itemFeatureQualityTable);

        doExplain = conf.getBoolean("rec.explain.flag");
        LOG.info("numUsers:" + numUsers);
        LOG.info("numItems:" + numItems);
        LOG.info("numFeatures:" + numberOfFeatures);
    }

    protected void explain(String userId) throws LibrecException {
        // get useridx and itemidices
        int userIdx = userMappingData.get(userId);
        double[] predRatings = new double[numItems];

        for (int itemIdx = 0; itemIdx < numItems; itemIdx++) {
            predRatings[itemIdx] = predictWithoutBound(userIdx, itemIdx);
        }

        // get the max\min predRating's index
        int maxIndex = 0;
        int minIndex = 0;
        for (int itemIdx = 0; itemIdx < numItems; itemIdx++) {
            double newnumber = predRatings[itemIdx];
            if (newnumber > predRatings[maxIndex]) {
                maxIndex = itemIdx;
            }
            if (newnumber < predRatings[minIndex]) {
                minIndex = itemIdx;
            }
        }

        int recommendedItemIdx = maxIndex;
        int disRecommendedItemIdx = minIndex;
        String recommendedItemId = itemMappingData.inverse().get(recommendedItemIdx);
        String disRecommendedItemId = itemMappingData.inverse().get(disRecommendedItemIdx);

        // get feature and values
        double[] userFeatureValues = featureMatrix.times(userFeatureMatrix.row(userIdx)).getValues();
        double[] recItemFeatureValues = featureMatrix.times(itemFeatureMatrix.row(recommendedItemIdx)).getValues();
        double[] disRecItemFeatureValues = featureMatrix.times(itemFeatureMatrix.row(disRecommendedItemIdx)).getValues();
        Map<Integer, Double> userFeatureValueMap = new HashMap<>();
        for (int i = 0; i < numberOfFeatures; i++) {
            userFeatureValueMap.put(i, userFeatureValues[i]);
        }
        // sort features by values

        userFeatureValueMap = sortByValue(userFeatureValueMap);
        //System.out.println("userFeatureValueMap");
        //System.out.println(userFeatureValueMap);

        // get top K feature and its values
        int numFeatureToExplain = conf.getInt("rec.explain.numfeature");
        Object[] userTopFeatureIndices = Arrays.copyOfRange(userFeatureValueMap.keySet().toArray(), numberOfFeatures - numFeatureToExplain, numberOfFeatures);
        String[] userTopFeatureIds = new String[numFeatureToExplain];
        double[] userTopFeatureValues = new double[numFeatureToExplain];
        double[] recItemTopFeatureValues = new double[numFeatureToExplain];
        double[] disRecItemTopFeatureIdValues = new double[numFeatureToExplain];
        for (int i = 0; i < numFeatureToExplain; i++) {
            int featureIdx = (int) userTopFeatureIndices[numFeatureToExplain - 1 - i];
            userTopFeatureValues[i] = userFeatureValues[featureIdx];
            recItemTopFeatureValues[i] = recItemFeatureValues[featureIdx];
            disRecItemTopFeatureIdValues[i] = disRecItemFeatureValues[featureIdx];
            userTopFeatureIds[i] = featureDict.inverse().get(featureIdx);
        }

        StringBuilder userFeatureSb = new StringBuilder();
        StringBuilder recItemFeatureSb = new StringBuilder();
        StringBuilder disRecItemFeatureSb = new StringBuilder();
        for (int i = 0; i < numFeatureToExplain; i++) {
            userFeatureSb.append(userTopFeatureIds[i]).append(":").append(normalize(userTopFeatureValues[i])).append("\n");
            recItemFeatureSb.append(userTopFeatureIds[i]).append(":").append(normalize(recItemTopFeatureValues[i])).append("\n");
            disRecItemFeatureSb.append(userTopFeatureIds[i]).append(":").append(normalize(disRecItemTopFeatureIdValues[i])).append("\n");
        }
        LOG.info("user " + userId + "'s most cared features are \n" + userFeatureSb);
        LOG.info("item " + recommendedItemId + "'s feature values are\n" + recItemFeatureSb);
        LOG.info("item " + disRecommendedItemId + "'s feature values are\n" + disRecItemFeatureSb);
        LOG.info("So we recommend item " + recommendedItemId + ", disRecommend item " + disRecommendedItemId + " to user " + userId);
        LOG.info("___________________________");
    }

    protected void trainModel() throws LibrecException {
        batchSize = conf.getInt("rec.sgd.batchSize", 100);
        epsilon = conf.getDouble("rec.sgd.epsilon", 1e-8);
        eta = conf.getDouble("rec.sgd.eta", 1.0);
        double[][] featureMatrixLearnRate = new double[numberOfFeatures][explicitFeatureNum];
        double[][] userFeatureMatrixLearnRate = new double[numUsers][explicitFeatureNum];
        double[][] userHiddenMatrixLearnRate = new double[numUsers][numFactors - explicitFeatureNum];
        double[][] itemFeatureMatrixLearnRate = new double[numItems][explicitFeatureNum];
        double[][] itemHiddenMatrixLearnRate = new double[numItems][numFactors - explicitFeatureNum];

        BatchSet userItemBatch = new BatchSet(conf, trainMatrix);
        BatchSet userFeatureBatch = new BatchSet(conf, userFeatureAttention);
        BatchSet itemFeatureBatch = new BatchSet(conf, itemFeatureQuality);

        for (int iter = 1; iter <= numIterations; iter++) {
            loss = 0.0d;
            int maxSampleSize = (trainMatrix.getNumEntries() + userFeatureAttention.getNumEntries() + itemFeatureQuality.getNumEntries()) /
                    (3 * batchSize);

            for (int sampleCount = 0; sampleCount < maxSampleSize; sampleCount++) {
                userItemBatch.sampling(batchSize);
                // user Feature Matrix(X) neg feature samples
                userFeatureBatch.sampling(batchSize);
                userFeatureBatch.rightNegSampling(batchSize);
                itemFeatureBatch.sampling(batchSize);
                //update hiddenUserFeatureMatrix
                Map<Integer, Set<Integer>> userItemsSet = userItemBatch.getLeftToRightSamples();
                for (Map.Entry userItems : userItemsSet.entrySet()) {
                    int userIdx = (Integer) userItems.getKey();
                    Set<Integer> itemsSet = (Set<Integer>) userItems.getValue();
                    Differential hiddenUserDiffs = new Differential(itemsSet);
                    hiddenUserDiffs.calcSetUp(hiddenFeatureNum);
                    for (differentialEntry hude : hiddenUserDiffs) {
                        int itemIdx = hude.get();
                        double ratingValue = trainMatrix.get(userIdx, itemIdx);
                        double predictValue = predictWithoutBound(userIdx, itemIdx);
                        DenseVector batchVector = itemHiddenMatrix.row(itemIdx);
                        hude.set(ratingValue, predictValue, batchVector);
                        loss += (ratingValue - predictValue) * (ratingValue - predictValue);
                    }
                    for (int factorIdx = 0; factorIdx < hiddenFeatureNum; factorIdx++) {
                        double realRatingValue = hiddenUserDiffs.getRealRatingValue(factorIdx);
                        double estmRatingValue = hiddenUserDiffs.getEstmRatingValue(factorIdx);
                        double userHiddenValue = userHiddenMatrix.get(userIdx, factorIdx);
                        double error = (realRatingValue - estmRatingValue) - lambdaH * userHiddenValue;
                        userHiddenMatrixLearnRate[userIdx][factorIdx] += error * error;
                        double del = adagrad(userHiddenMatrixLearnRate[userIdx][factorIdx], error, itemsSet.size());
                        userHiddenMatrix.plus(userIdx, factorIdx, del);
                        if (userHiddenMatrix.get(userIdx, factorIdx) < 0)
                            userHiddenMatrix.set(userIdx, factorIdx, 0.0);
                        loss += lambdaH * userHiddenValue * userHiddenValue;
                    }
                }
                //update hiddenItemFeatureMatrix
                Map<Integer, Set<Integer>> itemUsersSet = userItemBatch.getRightToLeftSamples();
                for (Map.Entry itemUsers : itemUsersSet.entrySet()) {
                    int itemIdx = (Integer) itemUsers.getKey();
                    Set<Integer> usersSet = (Set<Integer>) itemUsers.getValue();
                    Differential hiddenItemDiffs = new Differential(usersSet);
                    hiddenItemDiffs.calcSetUp(hiddenFeatureNum);
                    for (differentialEntry hide : hiddenItemDiffs) {
                        int userIdx = hide.get();
                        double ratingValue = trainMatrix.get(userIdx, itemIdx);
                        double predictValue = predictWithoutBound(userIdx, itemIdx);
                        DenseVector batchVector = userHiddenMatrix.row(userIdx);
                        hide.set(ratingValue, predictValue, batchVector);
                    }
                    for (int factorIdx = 0; factorIdx < hiddenFeatureNum; factorIdx++) {
                        double realRatingValue = hiddenItemDiffs.getRealRatingValue(factorIdx);
                        double estmRatingValue = hiddenItemDiffs.getEstmRatingValue(factorIdx);
                        double itemHiddenValue = itemHiddenMatrix.get(itemIdx, factorIdx);
                        double error = (realRatingValue - estmRatingValue)  - lambdaH * itemHiddenValue;
                        itemHiddenMatrixLearnRate[itemIdx][factorIdx] += error * error;
                        double del = adagrad(itemHiddenMatrixLearnRate[itemIdx][factorIdx], error, usersSet.size());
                        itemHiddenMatrix.plus(itemIdx, factorIdx, del);
                        if (itemHiddenMatrix.get(itemIdx, factorIdx) < 0)
                            itemHiddenMatrix.set(itemIdx, factorIdx, 0.0);
                        loss += lambdaH * itemHiddenValue * itemHiddenValue;
                    }
                }

                //update featureMatrix(V)
                Set<Integer> userPosNegFeatureUnion = userFeatureBatch.rightLogicalOR(userFeatureBatch.getRightNegUnion());
                Set<Integer> userItemFeatureUion = itemFeatureBatch.rightLogicalOR(userPosNegFeatureUnion);
                Map<Integer, Set<Integer>> featureUsersSet = userFeatureBatch.getRightToLeftSamples();
                Map<Integer, Set<Integer>> negFeatureUsersSet = userFeatureBatch.getNegRightToLeftSamples();
                Map<Integer, Set<Integer>> featureItemsSet = itemFeatureBatch.getRightToLeftSamples();
                for (Integer featureIdx : userItemFeatureUion) {
                    Differential userToFeatureDiffs, itemToFeatureDiffs;
                    Set<Integer> usersSet = new HashSet<>();
                    VectorBasedDenseVector deriValuesVector;
                    DenseMatrix batchUserMatrix;
                    if (featureUsersSet.containsKey(featureIdx)) {
                        usersSet = featureUsersSet.get(featureIdx);
                        deriValuesVector = new VectorBasedDenseVector(usersSet.size());
                        batchUserMatrix = new DenseMatrix(usersSet.size(), explicitFeatureNum);

                        int index = 0;
                        for (Integer userIdx : usersSet) {
                            int negFeatureIdx = userFeatureBatch.getLeftPosRightNegRightTable().get(userIdx, featureIdx);
                            double posPredictRating = predUserAttention(userIdx, featureIdx);
                            double negPredictRating = predUserAttention(userIdx, negFeatureIdx);
                            double diffValue = posPredictRating - negPredictRating;

                            double lossValue = -Math.log(Maths.logistic(diffValue));
                            loss += lossValue;
                            double deriValue = Maths.logistic(-diffValue);
                            deriValuesVector.set(index, deriValue);
                            batchUserMatrix.set(index, userFeatureMatrix.row(userIdx));
                        }

                    } else if (negFeatureUsersSet.containsKey(featureIdx)) {
                        usersSet = negFeatureUsersSet.get(featureIdx);
                        deriValuesVector = new VectorBasedDenseVector(usersSet.size());
                        batchUserMatrix = new DenseMatrix(usersSet.size(), explicitFeatureNum);

                        int index = 0;
                        for (Integer userIdx : usersSet) {
                            int posFeatureIdx = userFeatureBatch.getLeftNegRightPosRightTable().get(userIdx, featureIdx);
                            double posPredictRating = predUserAttention(userIdx, posFeatureIdx);
                            double negPredictRating = predUserAttention(userIdx, featureIdx);
                            double diffValue = posPredictRating - negPredictRating;

                            double lossValue = -Math.log(Maths.logistic(diffValue));
                            loss += lossValue;
                            double deriValue = Maths.logistic(-diffValue);
                            deriValuesVector.set(index, deriValue);
                            batchUserMatrix.set(index, userFeatureMatrix.row(userIdx));
                        }


                    } else {
                        deriValuesVector = new VectorBasedDenseVector(0);
                        batchUserMatrix = new DenseMatrix(0,0);
                    }
                    Set<Integer> itemsSet = new HashSet<>();
                    if (featureItemsSet.containsKey(featureIdx)) {
                        itemsSet = featureItemsSet.get(featureIdx);
                        itemToFeatureDiffs = new Differential(itemsSet);
                        itemToFeatureDiffs.calcSetUp(explicitFeatureNum);
                        for (differentialEntry ifde : itemToFeatureDiffs) {
                            int itemIdx = ifde.get();
                            double ratingValue = itemFeatureQuality.get(itemIdx, featureIdx);
                            double predictValue = predItemQuality(itemIdx, featureIdx);
                            DenseVector batchVector = itemFeatureMatrix.row(itemIdx);
                            ifde.set(ratingValue, predictValue, batchVector);
                            loss += (ratingValue - predictValue) * (ratingValue - predictValue);
                        }
                    } else {
                        itemToFeatureDiffs = new Differential(new HashSet<>());
                        itemToFeatureDiffs.calcSetUp(explicitFeatureNum);
                    }

                    for (int factorIdx = 0; factorIdx < explicitFeatureNum; factorIdx++) {
                        MatrixBasedDenseVector userFactorsVector = (MatrixBasedDenseVector) batchUserMatrix.column(factorIdx);
                        double featureFactorValue;
                        if (negFeatureUsersSet.containsKey(featureIdx)) {
                            featureFactorValue = -userFactorsVector.dot(deriValuesVector);
                        } else {
                            featureFactorValue = userFactorsVector.dot(deriValuesVector);
                        }
                        double realItemRatingValue = itemToFeatureDiffs.getRealRatingValue(factorIdx);
                        double estmItemRatingValue = itemToFeatureDiffs.getEstmRatingValue(factorIdx);
                        double featureValue = featureMatrix.get(featureIdx, factorIdx);
                        double error = lambdaX * featureFactorValue + lambdaY * (realItemRatingValue - estmItemRatingValue) - lambdaV * featureValue;
                        featureMatrixLearnRate[featureIdx][factorIdx] += error * error;
                        double del = adagrad(featureMatrixLearnRate[featureIdx][factorIdx], error, usersSet.size() + itemsSet.size());
                        featureMatrix.plus(featureIdx, factorIdx, del);
                        if (featureMatrix.get(featureIdx, factorIdx) < 0.0)
                            featureMatrix.set(featureIdx, factorIdx, 0.0);
                        loss += lambdaV * featureValue * featureValue;
                    }
                }

                //update userFeatureMatrix(U1)
                Set<Integer> itemUserUnion = userItemBatch.getLeftUnion();
                Set<Integer> usersUnion = userFeatureBatch.leftLogicalOR(itemUserUnion);
                Map<Integer, Set<Integer>> userFeaturesSet = userFeatureBatch.getLeftToRightSamples();
                Map<Integer, Set<Integer>> userNegFeaturesSet = userFeatureBatch.getLeftToRightNegSamples();
                for (Integer userIdx : usersUnion) {
                    Differential itemToUserDiffs, featureToUserDiffs;
                    Set<Integer> itemsSet = new HashSet<>();
                    if (userItemsSet.containsKey(userIdx)) {
                        itemsSet = userItemsSet.get(userIdx);
                        itemToUserDiffs = new Differential(itemsSet);
                        itemToUserDiffs.calcSetUp(explicitFeatureNum);
                        for (differentialEntry iude : itemToUserDiffs) {
                            int itemIdx = iude.get();
                            double ratingValue = trainMatrix.get(userIdx, itemIdx);
                            double predictValue = predictWithoutBound(userIdx, itemIdx);
                            DenseVector batchVector = itemFeatureMatrix.row(itemIdx);
                            iude.set(ratingValue, predictValue, batchVector);
                        }
                    } else {
                        itemToUserDiffs = new Differential(itemsSet);
                        itemToUserDiffs.calcSetUp(explicitFeatureNum);
                    }

                    Set<Integer> featuresSet = new HashSet<>();
                    List<Integer> negFeaturesList;
                    VectorBasedDenseVector deriValuesVector;
                    //feature matrix V
                    DenseMatrix batchPosFeatureMatrix, batchNegFeatureMatrix;
                    if (userFeaturesSet.containsKey(userIdx)) {
                        featuresSet = userFeaturesSet.get(userIdx);
                        negFeaturesList = new ArrayList<>(userNegFeaturesSet.get(userIdx));
                        deriValuesVector = new VectorBasedDenseVector(featuresSet.size());
                        batchPosFeatureMatrix = new DenseMatrix(featuresSet.size(), explicitFeatureNum);
                        batchNegFeatureMatrix = new DenseMatrix(featuresSet.size(), explicitFeatureNum);

                        int index = 0;
                        for (Integer posFeatureIdx : featuresSet) {
                           int negFeatureIdx =  negFeaturesList.get(index);
                           double posPredictRating = predUserAttention(userIdx, posFeatureIdx);
                           double negPredictRating = predUserAttention(userIdx, negFeatureIdx);
                           double diffValue = posPredictRating - negPredictRating;

                           double lossValue = -Math.log(Maths.logistic(diffValue));
                           loss += lossValue;
                           double deriValue = Maths.logistic(-diffValue);
                           deriValuesVector.set(index, deriValue);
                           batchPosFeatureMatrix.set(index, featureMatrix.row(posFeatureIdx));
                           batchNegFeatureMatrix.set(index, featureMatrix.row(negFeatureIdx));
                        }

                    } else {
                        deriValuesVector = new VectorBasedDenseVector(0);
                        batchPosFeatureMatrix = new DenseMatrix(0,0);
                        batchNegFeatureMatrix = new DenseMatrix(0,0);
                    }
                    for (int factorIdx = 0; factorIdx < explicitFeatureNum; factorIdx++) {
                        double realItemRatingValue = itemToUserDiffs.getRealRatingValue(factorIdx);
                        double estmItemRatingValue = itemToUserDiffs.getEstmRatingValue(factorIdx);

                        MatrixBasedDenseVector posFeatureFactorsVector = (MatrixBasedDenseVector) batchPosFeatureMatrix.column(factorIdx);
                        MatrixBasedDenseVector negFeatureFactorsVector = (MatrixBasedDenseVector) batchNegFeatureMatrix.column(factorIdx);
                        double posFeatureFactorValue = posFeatureFactorsVector.dot(deriValuesVector);
                        double negFeatureFactorValue = negFeatureFactorsVector.dot(deriValuesVector);

                        double userFeatureValue = userFeatureMatrix.get(userIdx, factorIdx);
                        double error = (realItemRatingValue - estmItemRatingValue) + lambdaX * (posFeatureFactorValue - negFeatureFactorValue)
                                - lambdaU * userFeatureValue;
                        userFeatureMatrixLearnRate[userIdx][factorIdx] += error * error;
                        double del = adagrad(userFeatureMatrixLearnRate[userIdx][factorIdx], error, itemsSet.size() + featuresSet.size());
                        userFeatureMatrix.plus(userIdx, factorIdx, del);
                        if (userFeatureMatrix.get(userIdx, factorIdx) < 0.0)
                            userFeatureMatrix.set(userIdx, factorIdx, 0.0);
                    }
                }

                //update itemFeatureMatrix(U2)
                Set<Integer> userToItemUnion = userItemBatch.getRightUnion();
                Set<Integer> itemUnion = itemFeatureBatch.leftLogicalOR(userToItemUnion);
                Map<Integer, Set<Integer>> itemFeaturesSet = itemFeatureBatch.getLeftToRightSamples();
                for (Integer itemIdx : itemUnion) {
                    Differential userToItemDiffs, featureToItemDiffs;
                    Set<Integer> usersSet = new HashSet<>();
                    if (itemUsersSet.containsKey(itemIdx)) {
                        usersSet = itemUsersSet.get(itemIdx);
                        userToItemDiffs = new Differential(usersSet);
                        userToItemDiffs.calcSetUp(explicitFeatureNum);
                        for (differentialEntry uide : userToItemDiffs) {
                            int userIdx = uide.get();
                            double ratingValue = trainMatrix.get(userIdx, itemIdx);
                            double predictValue = predictWithoutBound(userIdx, itemIdx);
                            DenseVector batchVector = userFeatureMatrix.row(userIdx);
                            uide.set(ratingValue, predictValue, batchVector);
                        }
                    } else {
                        userToItemDiffs = new Differential(usersSet);
                        userToItemDiffs.calcSetUp(explicitFeatureNum);
                    }

                    Set<Integer> featuresSet = new HashSet<>();
                    if (itemFeaturesSet.containsKey(itemIdx)) {
                        featuresSet = itemFeaturesSet.get(itemIdx);
                        featureToItemDiffs = new Differential(featuresSet);
                        featureToItemDiffs.calcSetUp(explicitFeatureNum);
                        for (differentialEntry fide : featureToItemDiffs) {
                            int featureIdx = fide.get();
                            double ratingValue = itemFeatureQuality.get(itemIdx, featureIdx);
                            double predictValue = predItemQuality(itemIdx, featureIdx);
                            DenseVector batchVector = featureMatrix.row(featureIdx);
                            fide.set(ratingValue, predictValue, batchVector);
                        }
                    } else {
                        featureToItemDiffs = new Differential(featuresSet);
                        featureToItemDiffs.calcSetUp(explicitFeatureNum);
                    }
                    for (int factorIdx = 0; factorIdx < explicitFeatureNum; factorIdx++) {
                        double realUserRatingValue =userToItemDiffs.getRealRatingValue(factorIdx);
                        double estmUserRatingValue =userToItemDiffs.getEstmRatingValue(factorIdx);
                        double realFeatureRatingValue = featureToItemDiffs.getRealRatingValue(factorIdx);
                        double estmFeatureRatingValue = featureToItemDiffs.getEstmRatingValue(factorIdx);
                        double itemFeatureValue = itemFeatureMatrix.get(itemIdx, factorIdx);
                        double error = (realUserRatingValue - estmUserRatingValue) + lambdaY * (realFeatureRatingValue - estmFeatureRatingValue)
                                - lambdaU * itemFeatureValue;
                        itemFeatureMatrixLearnRate[itemIdx][factorIdx] += error * error;
                        double del = adagrad(itemFeatureMatrixLearnRate[itemIdx][factorIdx], error, usersSet.size() + featuresSet.size());
                        itemFeatureMatrix.plus(itemIdx, factorIdx, del);
                        if (itemFeatureMatrix.get(itemIdx, factorIdx) < 0.0)
                            itemFeatureMatrix.set(itemIdx, factorIdx, 0.0);
                    }
                }
            }
            LOG.info("iter:" + iter + ", loss:" + loss);
        }
        maxUserFeature = -100000;
        minUserFeature = 1000000;
        for (int userIdx = 0; userIdx < numUsers; userIdx++) {
            for (int featureIdx = 0; featureIdx < numberOfFeatures; featureIdx++) {
                double predValue = predUserAttention(userIdx, featureIdx);
                maxUserFeature = Math.max(predValue, maxUserFeature);
                minUserFeature = Math.min(predValue, minUserFeature);
            }
        }
    }

    @Override
    protected double predict(int[] indices) {
        return predict(indices[0], indices[1]);
    }

    protected double predict(int u, int j) {
        doRanking = conf.getBoolean("rec.recommend.doRanking", false);
        if (doRanking && isRanking) {
            double pred = topKPredict(u,j);
            return pred;
        } else {
            double pred = userFeatureMatrix.row(u).dot(itemFeatureMatrix.row(j)) + userHiddenMatrix.row(u).dot(itemHiddenMatrix.row(j));
            if (pred < minRate)
                return minRate;
            if (pred > maxRate)
                return maxRate;
            return pred;
        }
    }

    protected double topKPredict(int u, int j) {
        double tradeoff = conf.getDouble("rec.tradeoff", 0.0);
        double pred = (1.0 - tradeoff) * predictWithoutBound(u, j);

        double[] userFeatureValues = new double[numberOfFeatures];
        double[] recItemFeatureValues = new double[numberOfFeatures];
        if (userFeatureValuesVectorList.get(u) == null) {
            userFeatureValues = featureMatrix.times(userFeatureMatrix.row(u)).getValues();
            VectorBasedDenseVector userFeatureValuesVector = new VectorBasedDenseVector(userFeatureValues);
            userFeatureValuesVectorList.set(u, userFeatureValuesVector);
        } else {
            userFeatureValues = userFeatureValuesVectorList.get(u).getValues();
        }
        if (recItemFeatureValuesVectorList.get(j) == null) {
            recItemFeatureValues = featureMatrix.times(itemFeatureMatrix.row(j)).getValues();
            VectorBasedDenseVector recItemFeatureValuesVector = new VectorBasedDenseVector(recItemFeatureValues);
            recItemFeatureValuesVectorList.set(j, recItemFeatureValuesVector);
        } else {
            recItemFeatureValues = recItemFeatureValuesVectorList.get(j).getValues();

        }

        //userFeatureValues = featureMatrix.times(userFeatureMatrix.row(u)).getValues();
        //recItemFeatureValues = featureMatrix.times(itemFeatureMatrix.row(j)).getValues();
        Map<Integer, Double> userFeatureValueMap = new HashMap<>();
        if (userFeatureValuesList.get(u) == null) {
            for (int i = 0; i < numberOfFeatures; i++) {
                userFeatureValueMap.put(i, userFeatureValues[i]);
            }
            // sort features by values
            userFeatureValueMap = sortByValue(userFeatureValueMap);
            userFeatureValuesList.set(u, userFeatureValueMap);
        } else {
            userFeatureValueMap = userFeatureValuesList.get(u);
        }
        int numFeatureTopK = conf.getInt("rec.recommend.numfeatureTopK", 10);
        Object[] userTopFeatureIndices = Arrays.copyOfRange(userFeatureValueMap.keySet().toArray(), numberOfFeatures - numFeatureTopK, numberOfFeatures);
        double[] userTopFeatureValues = new double[numFeatureTopK];
        double[] recItemTopFeatureValues = new double[numFeatureTopK];
        //need to normalize because feature score is not range (minRate, maxRate) in bpr algorithm
        for (int i=0; i<numFeatureTopK; i++) {
            int featureIdx = (int) userTopFeatureIndices[numFeatureTopK - 1 - i];
            userTopFeatureValues[i] = normalize(maxUserFeature, minUserFeature, userFeatureValues[featureIdx]);
            recItemTopFeatureValues[i] = normalize(recItemFeatureValues[featureIdx]);
        }

        //calc k largest values
        double largestValue = 0.0;
        for (int i = 0; i < numFeatureTopK; i++) {
            largestValue += userTopFeatureValues[i] * recItemTopFeatureValues[i];
        }
        //rescale to (minRate, maxRate)
        largestValue *= maxRate * (tradeoff / numFeatureTopK);
        pred += largestValue;

        return pred;

    }

    protected double predictWithoutBound(int u, int j) {
        return userFeatureMatrix.row(u).dot(itemFeatureMatrix.row(j)) + userHiddenMatrix.row(u).dot(itemHiddenMatrix.row(j));
    }

    protected double predUserAttention(int userIdx, int featureIdx) {
        return userFeatureMatrix.row(userIdx).dot(featureMatrix.row(featureIdx));
    }

    protected double predItemQuality(int itemIdx, int featureIdx) {
        return itemFeatureMatrix.row(itemIdx).dot(featureMatrix.row(featureIdx));
    }

    /**
     * Sort a map by value.
     *
     * @param map the map to sort
     * @param <K> key type
     * @param <V> value type
     * @return a sorted map of the input
     */
    protected static <K, V extends Comparable<? super V>> Map<K, V> sortByValue(Map<K, V> map) {
        List<Map.Entry<K, V>> list = new LinkedList<>(map.entrySet());
        Collections.sort(list, new Comparator<Map.Entry<K, V>>() {
            public int compare(Map.Entry<K, V> o1, Map.Entry<K, V> o2) {
                return (o1.getValue()).compareTo(o2.getValue());
            }
        });

        Map<K, V> result = new LinkedHashMap<K, V>();
        for (Map.Entry<K, V> entry : list) {
            result.put(entry.getKey(), entry.getValue());
        }
        return result;
    }

    /**
     * Normalize the value into [0, 1]
     *
     * @param rating the input value
     * @return the normalized value
     */
    protected double normalize(double rating) {
        return (rating - minRate) / (maxRate - minRate);
    }

    protected double normalize(double max, double min, double value) {
        return (value - min) / (max - min);
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

    protected static class Differential implements Iterable<differentialEntry> {
        Set<Integer> batchSet;
        VectorBasedDenseVector ratingsVector;
        VectorBasedDenseVector predictsVector;
        DenseMatrix batchMatrix;
        Set<Integer> union;
        int setSize;
        Integer[] keyIndices;

        public Differential(Set<Integer> batchSet) {
            this.batchSet = batchSet;
            keyIndices = batchSet.toArray(new Integer[batchSet.size()]);
        }


        public void calcSetUp(int numFactors) {
            setSize = batchSet.size();
            ratingsVector = new VectorBasedDenseVector(setSize);
            predictsVector = new VectorBasedDenseVector(setSize);
            if (setSize == 0) {
                batchMatrix = new DenseMatrix(0, 0);
            } else {
                batchMatrix = new DenseMatrix(setSize, numFactors);
            }
        }

        public void setValue(int idx, double ratingValue, double predictValue, DenseVector batchVector) {
            ratingsVector.set(idx, ratingValue);
            predictsVector.set(idx, predictValue);
            batchMatrix.set(idx, batchVector);
        }

        public DenseMatrix getBatchMatrix() {
            return batchMatrix;
        }

        double getRealRatingValue(int factorIdx) {
            return ratingsVector.dot(batchMatrix.column(factorIdx));
        }

        double getEstmRatingValue(int factorIdx) {
            return predictsVector.dot(batchMatrix.column(factorIdx));
        }

        private class DifferentialIterator implements Iterator<differentialEntry> {
            private int index = 0;
            private DifferentialEntry entry = new DifferentialEntry();

            public boolean hasNext() {
                return index < batchSet.size();
            }

            public differentialEntry next() {
                return entry.update(index++);
            }
        }

        private class DifferentialEntry implements differentialEntry {
            private int index = -1;

            public DifferentialEntry update(int index) {
                this.index = index;
                return this;
            }

            public int key() {
                return keyIndices[index];
            }


            public Integer[] keys() {
                return keyIndices;
            }

            public int get() {
                return keyIndices[index];
            }

            public void set(double ratingValue, double predictValue, DenseVector batchVector) {
                ratingsVector.set(this.index, ratingValue);
                predictsVector.set(this.index, predictValue);
                batchMatrix.set(this.index, batchVector);
            }
        }

        public Iterator<differentialEntry> iterator() {
            return new DifferentialIterator();
        }
    }

    public interface differentialEntry {
        Integer[] keys();

        int get();

        void set(double ratingValue, double predictValue, DenseVector batchVector);
    }


}
