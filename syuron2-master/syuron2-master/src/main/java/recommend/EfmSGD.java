package recommend;

import com.google.common.collect.BiMap;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.HashBiMap;
import com.google.common.collect.Table;
import net.librec.common.LibrecException;
import net.librec.math.algorithm.Randoms;
import net.librec.math.structure.*;
import net.librec.math.structure.Vector;
import net.librec.recommender.TensorRecommender;
import org.apache.commons.lang.StringUtils;

import java.util.*;

public class EfmSGD extends TensorRecommender{
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

    /*
     * (non-Javadoc)
     *
     * @see net.librec.recommender.AbstractRecommender#setup()
     */
    @Override
    protected void setup() throws LibrecException {
        super.setup();
        batchSize = conf.getInt("rec.sgd.batchSize", 100);
        epsilon = conf.getDouble("rec.sgd.epsilon", 1e-8);
        eta = conf.getDouble("rec.sgd.eta", 1.0);
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

    //@Override
    /**
    protected void trainModel() throws LibrecException {
        for (int iter = 1; iter <= conf.getInt("rec.iterator.maximum"); iter++) {
            loss = 0.0;
            updateProgress(0);
            // Update featureMatrix by fixing the others
            // LOG.info("iter:" + iter + ", Update featureMatrix");
            for(int featureIdx=0; featureIdx<numberOfFeatures; featureIdx++) {
                SequentialSparseVector attentionVec = userFeatureAttention.column(featureIdx);
                SequentialSparseVector qualityVec = itemFeatureQuality.column(featureIdx);
                if (attentionVec.getNumEntries() > 0 && qualityVec.getNumEntries() > 0) {
                    RandomAccessSparseVector attentionPredVec = new RandomAccessSparseVector(numUsers);
                    RandomAccessSparseVector qualityPredVec = new RandomAccessSparseVector(numItems);

                    for (int userIdx: attentionVec.getIndices()) {
                        attentionPredVec.set(userIdx, predUserAttention(userIdx, featureIdx));
                    }
                    for (int itemIdx: qualityVec.getIndices()) {
                        qualityPredVec.set(itemIdx, predItemQuality(itemIdx, featureIdx));
                    }

                    for (int factorIdx=0; factorIdx<explicitFeatureNum; factorIdx++) {
                        DenseVector factorUsersVector = userFeatureMatrix.column(factorIdx);
                        DenseVector factorItemsVector = itemFeatureMatrix.column(factorIdx);

                        double numerator = lambdaX * factorUsersVector.dot(attentionVec) + lambdaY * factorItemsVector.dot(qualityVec);
                        double denominator = lambdaX * factorUsersVector.dot(attentionPredVec) + lambdaY * factorItemsVector.dot(qualityPredVec)
                                + lambdaV * featureMatrix.get(featureIdx, factorIdx) + 1e-9;

                        featureMatrix.set(featureIdx, factorIdx, featureMatrix.get(featureIdx, factorIdx) * Math.sqrt(numerator/denominator));

                    }
                }
            }
            updateProgress(20);


            // Update UserFeatureMatrix by fixing the others
            for (int userIdx=0; userIdx<numUsers; userIdx++) {
                SequentialSparseVector itemRatingsVector = trainMatrix.row(userIdx);
                SequentialSparseVector attentionVec = userFeatureAttention.row(userIdx);

                if (itemRatingsVector.getNumEntries() > 0 && attentionVec.getNumEntries() > 0) {
                    RandomAccessSparseVector itemPredictsVector = new RandomAccessSparseVector(numItems);
                    RandomAccessSparseVector attentionPredVec = new RandomAccessSparseVector(numberOfFeatures);

                    for (int itemIdx : itemRatingsVector.getIndices()) {
                        itemPredictsVector.set(itemIdx, predictWithoutBound(userIdx, itemIdx));
                    }

                    for (int featureIdx: attentionVec.getIndices()) {
                        attentionPredVec.set(featureIdx, predUserAttention(userIdx, featureIdx));
                    }

                    for (int factorIdx = 0; factorIdx < explicitFeatureNum; factorIdx++) {
                        DenseVector factorItemsVector = itemFeatureMatrix.column(factorIdx);
                        DenseVector featureVector = featureMatrix.column(factorIdx);

                        double numerator = factorItemsVector.dot(itemRatingsVector) + lambdaX * featureVector.dot(attentionVec);
                        double denominator = factorItemsVector.dot(itemPredictsVector) + lambdaX * featureVector.dot(attentionPredVec)
                                + lambdaU * userFeatureMatrix.get(userIdx, factorIdx) + 1e-9;

                        userFeatureMatrix.set(userIdx, factorIdx, userFeatureMatrix.get(userIdx, factorIdx) * Math.sqrt(numerator/denominator));
                    }
                }
            }
            updateProgress(40);

            // Update ItemFeatureMatrix by fixing the others
            // LOG.info("iter:" + iter + ", Update ItemFeatureMatrix");
            for (int itemIdx = 0; itemIdx < numItems; itemIdx++) {
                SequentialSparseVector userRatingsVector = trainMatrix.column(itemIdx);
                SequentialSparseVector qualityVector = itemFeatureQuality.row(itemIdx);

                if (userRatingsVector.getNumEntries() > 0 && qualityVector.getNumEntries() > 0) {
                    RandomAccessSparseVector userPredictsVector = new RandomAccessSparseVector(numUsers);
                    RandomAccessSparseVector qualityPredVec = new RandomAccessSparseVector(numberOfFeatures);

                    for (int userIdx : userRatingsVector.getIndices()) {
                        userPredictsVector.set(userIdx, predictWithoutBound(userIdx, itemIdx));
                    }

                    for (int featureIdx : qualityVector.getIndices()) {
                        qualityPredVec.set(featureIdx, predItemQuality(itemIdx, featureIdx));
                    }

                    for (int factorIdx = 0; factorIdx < explicitFeatureNum; factorIdx++) {
                        DenseVector factorUsersVector = userFeatureMatrix.column(factorIdx);
                        DenseVector featureVector = featureMatrix.column(factorIdx);

                        double numerator = factorUsersVector.dot(userRatingsVector) + lambdaY * featureVector.dot(qualityVector);
                        double denominator = factorUsersVector.dot(userPredictsVector) + lambdaY * featureVector.dot(qualityPredVec)
                                + lambdaU * itemFeatureMatrix.get(itemIdx, factorIdx) + 1e-9;

                        itemFeatureMatrix.set(itemIdx, factorIdx, itemFeatureMatrix.get(itemIdx, factorIdx) * Math.sqrt(numerator/denominator));
                    }
                }
            }
            updateProgress(60);

            // Update UserHiddenMatrix by fixing the others
            // LOG.info("iter:" + iter + ", Update UserHiddenMatrix");
            for (int userIdx=0; userIdx<numUsers; userIdx++) {
                SequentialSparseVector itemRatingsVector = trainMatrix.row(userIdx);
                if (itemRatingsVector.getNumEntries() > 0) {
                    RandomAccessSparseVector itemPredictsVector = new RandomAccessSparseVector(numItems);

                    for (int itemIdx : itemRatingsVector.getIndices()) {
                        itemPredictsVector.set(itemIdx, predictWithoutBound(userIdx, itemIdx));
                    }

                    for (int factorIdx = 0; factorIdx < hiddenFeatureNum; factorIdx++) {
                        DenseVector hiddenItemsVector = itemHiddenMatrix.column(factorIdx);
                        double numerator = hiddenItemsVector.dot(itemRatingsVector);
                        double denominator = hiddenItemsVector.dot(itemPredictsVector) + lambdaH * userHiddenMatrix.get(userIdx, factorIdx) + 1e-9;
                        userHiddenMatrix.set(userIdx, factorIdx, userHiddenMatrix.get(userIdx, factorIdx) * Math.sqrt(numerator/denominator));
                    }
                }
            }
            updateProgress(90);

            // Update ItemHiddenMatrix by fixing the others
            // LOG.info("iter:" + iter + ", Update ItemHiddenMatrix");
            for (int itemIdx=0; itemIdx<numItems; itemIdx++) {
                SequentialSparseVector userRatingsVector = trainMatrix.column(itemIdx);
                if (userRatingsVector.getNumEntries() > 0) {
                    RandomAccessSparseVector userPredictsVector = new RandomAccessSparseVector(numUsers);

                    for (int userIdx : userRatingsVector.getIndices()) {
                        userPredictsVector.set(userIdx, predictWithoutBound(userIdx, itemIdx));
                    }

                    for (int factorIdx = 0; factorIdx < hiddenFeatureNum; factorIdx++) {
                        DenseVector hiddenUsersVector = userHiddenMatrix.column(factorIdx);
                        double numerator = hiddenUsersVector.dot(userRatingsVector);
                        double denominator = hiddenUsersVector.dot(userPredictsVector) + lambdaH * itemHiddenMatrix.get(itemIdx, factorIdx) + 1e-9;
                        itemHiddenMatrix.set(itemIdx, factorIdx, itemHiddenMatrix.get(itemIdx, factorIdx) * Math.sqrt(numerator/denominator));
                    }
                }
            }
            updateProgress(100);

            // Compute loss value
            for (MatrixEntry me: trainMatrix) {
                int userIdx = me.row();
                int itemIdx = me.column();
                double rating = me.get();
                double predRating = predictWithoutBound(userIdx, itemIdx);
                loss += (rating - predRating) * (rating - predRating);
            }

            for (MatrixEntry me: userFeatureAttention) {
                int userIdx = me.row();
                int featureIdx = me.column();
                double real = me.get();
                double pred = predUserAttention(userIdx, featureIdx);
                loss += (real - pred) * (real - pred);
            }

            for (MatrixEntry me: itemFeatureQuality) {
                int itemIdx = me.row();
                int featureIdx = me.column();
                double real = me.get();
                double pred = predItemQuality(itemIdx, featureIdx);
                loss += (real - pred) * (real - pred);
            }

            loss += lambdaU * (Math.pow(userFeatureMatrix.norm(), 2) + Math.pow(itemFeatureMatrix.norm(), 2));
            loss += lambdaH * (Math.pow(userHiddenMatrix.norm(), 2) + Math.pow(itemHiddenMatrix.norm(), 2));
            loss += lambdaV * Math.pow(featureMatrix.norm(), 2);

            LOG.info("iter:" + iter + ", loss:" + loss);
        }
        if (doExplain) {
            String[] userIds = conf.get("rec.explain.userids").split(" ");
            for (String userId: userIds) {
                explain(userId);
            }
        }
    }
     */

    protected void explain(String userId) throws LibrecException {
        // get useridx and itemidices
        int userIdx = userMappingData.get(userId);
        double[] predRatings = new double[numItems];

        for (int itemIdx=0; itemIdx<numItems; itemIdx++) {
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
        double [] recItemFeatureValues = featureMatrix.times(itemFeatureMatrix.row(recommendedItemIdx)).getValues();
        double [] disRecItemFeatureValues = featureMatrix.times(itemFeatureMatrix.row(disRecommendedItemIdx)).getValues();
        Map<Integer, Double> userFeatureValueMap = new HashMap<>();
        for (int i=0; i<numberOfFeatures; i++) {
            userFeatureValueMap.put(i, userFeatureValues[i]);
        }
        // sort features by values
        userFeatureValueMap = sortByValue(userFeatureValueMap);

        // get top K feature and its values
        int numFeatureToExplain = conf.getInt("rec.explain.numfeature");
        Object[] userTopFeatureIndices = Arrays.copyOfRange(userFeatureValueMap.keySet().toArray(), numberOfFeatures - numFeatureToExplain, numberOfFeatures);
        String[] userTopFeatureIds = new String[numFeatureToExplain];
        double[] userTopFeatureValues = new double[numFeatureToExplain];
        double[] recItemTopFeatureValues = new double[numFeatureToExplain];
        double[] disRecItemTopFeatureIdValues = new double[numFeatureToExplain];
        for (int i=0; i<numFeatureToExplain; i++) {
            int featureIdx = (int) userTopFeatureIndices[numFeatureToExplain - 1 - i];
            userTopFeatureValues[i] = userFeatureValues[featureIdx];
            recItemTopFeatureValues[i] = recItemFeatureValues[featureIdx];
            disRecItemTopFeatureIdValues[i] = disRecItemFeatureValues[featureIdx];
            userTopFeatureIds[i] = featureDict.inverse().get(featureIdx);
        }

        StringBuilder userFeatureSb = new StringBuilder();
        StringBuilder recItemFeatureSb = new StringBuilder();
        StringBuilder disRecItemFeatureSb = new StringBuilder();
        for (int i=0; i<numFeatureToExplain; i++) {
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
    protected void trainModel() throws LibrecException{
        double[][] featureMatrixLearnRate = new double[numberOfFeatures][explicitFeatureNum];
        double[][] userFeatureMatrixLearnRate = new double[numUsers][explicitFeatureNum];
        double[][] userHiddenMatrixLearnRate = new double[numUsers][numFactors - explicitFeatureNum];
        double[][] itemFeatureMatrixLearnRate = new double[numItems][explicitFeatureNum];
        double[][] itemHiddenMatrixLearnRate = new double[numItems][numFactors - explicitFeatureNum];

        List<Set<Integer>> trainUserItemsSet = getRowColumnsSet(trainMatrix, numUsers);
        List<Set<Integer>> trainUserFeaturesSet = getRowColumnsSet(userFeatureAttention, numUsers);
        List<Set<Integer>> trainItemFeaturesSet = getRowColumnsSet(itemFeatureQuality, numItems);

        for (int iter = 1; iter <= numIterations; iter++) {
            loss = 0.0d;
            int maxSampleSize = (trainMatrix.getNumEntries() + userFeatureAttention.getNumEntries() + itemFeatureQuality.getNumEntries()) /
                    (3 * batchSize);

            for (int sampleCount = 0; sampleCount < maxSampleSize; sampleCount++) {
                Map<Integer, Set<Integer>> userItemsSet = new HashMap<>(numUsers * 4/3 + 1);
                Map<Integer, Set<Integer>> itemUsersSet = new HashMap<>(numItems * 4/3 + 1);
                Map<Integer, Set<Integer>> userFeaturesSet = new HashMap<>(numUsers * 4/3 + 1);
                Map<Integer, Set<Integer>> featureUsersSet = new HashMap<>(numberOfFeatures * 4/3 + 1);
                Map<Integer, Set<Integer>> itemFeaturesSet = new HashMap<>(numItems * 4/3 + 1);
                Map<Integer, Set<Integer>> featureItemsSet = new HashMap<>(numberOfFeatures * 4/3 + 1);
                Set<Integer> usersUnion = new HashSet<>(numUsers);
                Set<Integer> itemsUnion = new HashSet<>(numItems);
                Set<Integer> featuresUnion = new HashSet<>(numberOfFeatures);

                int userItemSample = 0;
                int userFeatureSample = 0;
                int itemFeatureSample = 0;

                while (userItemSample < batchSize) {
                    int userIdx = Randoms.uniform(numUsers);
                    Set<Integer> itemSet = trainUserItemsSet.get(userIdx);
                    if (itemSet.size() == 0 || itemSet.size() == numItems)
                        continue;
                    int[] itemIndices = trainMatrix.row(userIdx).getIndices();
                    int itemIdx = itemIndices[Randoms.uniform(itemIndices.length)];

                    if (!userItemsSet.containsKey(userIdx)) {
                        Set<Integer> batchItemsSet = new HashSet<>();
                        batchItemsSet.add(itemIdx);
                        userItemsSet.put(userIdx, batchItemsSet);
                    } else if (!userItemsSet.get(userIdx).contains(itemIdx)) {
                        Set<Integer> batchItemsSet = userItemsSet.get(userIdx);
                        batchItemsSet.add(itemIdx);
                    } else {
                        continue;
                    }

                    if (!itemUsersSet.containsKey(itemIdx)) {
                        Set<Integer> batchUsersSet = new HashSet<>();
                        batchUsersSet.add(userIdx);
                        itemUsersSet.put(itemIdx, batchUsersSet);
                    } else {
                        Set<Integer> batchUsersSet = itemUsersSet.get(itemIdx);
                        batchUsersSet.add(userIdx);
                    }
                    usersUnion.add(userIdx);
                    itemsUnion.add(itemIdx);
                    userItemSample++;
                }

                while (userFeatureSample < batchSize) {
                    int userIdx = Randoms.uniform(numUsers);
                    Set<Integer> featureSet = trainUserFeaturesSet.get(userIdx);
                    if (featureSet.size() == 0 || featureSet.size() == numberOfFeatures)
                        continue;

                    int[] featureIndices = userFeatureAttention.row(userIdx).getIndices();
                    int featureIdx = featureIndices[Randoms.uniform(featureIndices.length)];
                    if (!userFeaturesSet.containsKey(userIdx)) {
                        Set<Integer> batchFeaturesSet = new HashSet<>();
                        batchFeaturesSet.add(featureIdx);
                        userFeaturesSet.put(userIdx, batchFeaturesSet);
                    } else if (!userFeaturesSet.get(userIdx).contains(featureIdx)) {
                        Set<Integer> batchFeaturesSet = userFeaturesSet.get(userIdx);
                        batchFeaturesSet.add(featureIdx);
                    } else {
                        continue;
                    }

                    if (!featureUsersSet.containsKey(featureIdx)) {
                        Set<Integer> batchUsersSet = new HashSet<>();
                        batchUsersSet.add(userIdx);
                        featureUsersSet.put(featureIdx, batchUsersSet);
                    } else {
                        Set<Integer> batchUsersSet = featureUsersSet.get(featureIdx);
                        batchUsersSet.add(userIdx);
                    }
                    usersUnion.add(userIdx);
                    featuresUnion.add(featureIdx);
                    userFeatureSample++;
                }

                while (itemFeatureSample < batchSize) {
                    int itemIdx = Randoms.uniform(numItems);
                    Set<Integer> featureSet = trainItemFeaturesSet.get(itemIdx);
                    if (featureSet.size() == 0 || featureSet.size() == numberOfFeatures)
                        continue;

                    int[] featureIndices = itemFeatureQuality.row(itemIdx).getIndices();
                    int featureIdx = featureIndices[Randoms.uniform(featureIndices.length)];
                    if (!itemFeaturesSet.containsKey(itemIdx)) {
                        Set<Integer> batchFeaturesSet = new HashSet<>();
                        batchFeaturesSet.add(featureIdx);
                        itemFeaturesSet.put(itemIdx, batchFeaturesSet);
                    } else if (!itemFeaturesSet.get(itemIdx).contains(featureIdx)) {
                        Set<Integer> batchFeaturesSet = itemFeaturesSet.get(itemIdx);
                        batchFeaturesSet.add(featureIdx);
                    } else {
                        continue;
                    }

                    if (!featureItemsSet.containsKey(featureIdx)) {
                        Set<Integer> batchItemsSet = new HashSet<>();
                        batchItemsSet.add(itemIdx);
                        featureItemsSet.put(featureIdx, batchItemsSet);
                    } else {
                        Set<Integer> batchItemsSet = featureItemsSet.get(featureIdx);
                        batchItemsSet.add(itemIdx);
                    }
                    itemsUnion.add(itemIdx);
                    featuresUnion.add(featureIdx);
                    itemFeatureSample++;
                }

                //update hiddenUserFeatureMatrix
                for (Map.Entry userItemsEntry : userItemsSet.entrySet()) {
                    int userIdx = (Integer) userItemsEntry.getKey();
                    int userItemsEntrySize = ((Set<Integer>) userItemsEntry.getValue()).size();
                    VectorBasedDenseVector itemRatingsVector = new VectorBasedDenseVector(userItemsEntrySize);
                    VectorBasedDenseVector itemPredictsVector = new VectorBasedDenseVector(userItemsEntrySize);
                    DenseMatrix batchItemHiddenMatrix = new DenseMatrix(userItemsEntrySize, hiddenFeatureNum);

                    int index = 0;
                    for (Integer itemIdx : (Set<Integer>) userItemsEntry.getValue()) {
                        itemRatingsVector.set(index, trainMatrix.get(userIdx, itemIdx));
                        itemPredictsVector.set(index, predictWithoutBound(userIdx, itemIdx));
                        batchItemHiddenMatrix.set(index, itemHiddenMatrix.row(itemIdx));
                        double lossError = (trainMatrix.get(userIdx, itemIdx) - predictWithoutBound(userIdx, itemIdx));
                        loss += lossError * lossError;
                        index++;
                    }

                    for (int factorIdx = 0; factorIdx < hiddenFeatureNum; factorIdx++) {
                        MatrixBasedDenseVector hiddenItemsVector = (MatrixBasedDenseVector) batchItemHiddenMatrix.column(factorIdx);
                        double realRatingValue = hiddenItemsVector.dot(itemRatingsVector);
                        double estmRatingValue = hiddenItemsVector.dot(itemPredictsVector);
                        double userHiddenValue = userHiddenMatrix.get(userIdx, factorIdx);
                        double error = (realRatingValue - estmRatingValue) - lambdaH * userHiddenValue;

                        userHiddenMatrixLearnRate[userIdx][factorIdx] += error * error;
                        double del = adagrad(userHiddenMatrixLearnRate[userIdx][factorIdx], error, userItemsEntrySize);
                        userHiddenMatrix.plus(userIdx, factorIdx, del);
                        if (userHiddenMatrix.get(userIdx, factorIdx) < 0)
                            userHiddenMatrix.set(userIdx, factorIdx, 0.0);
                        loss += lambdaH * userHiddenValue * userHiddenValue;
                    }
                }

                //update hiddenItemFeatureMatrix
                for (Map.Entry itemUsersEntry : itemUsersSet.entrySet()) {
                    int itemIdx = (Integer) itemUsersEntry.getKey();
                    int itemUsersEntrySize = ((Set<Integer>) itemUsersEntry.getValue()).size();
                    VectorBasedDenseVector userRatingsVector = new VectorBasedDenseVector(itemUsersEntrySize);
                    VectorBasedDenseVector userPredictsVector = new VectorBasedDenseVector(itemUsersEntrySize);
                    DenseMatrix batchUserHiddenMatrix = new DenseMatrix(itemUsersEntrySize, hiddenFeatureNum);

                    int index = 0;
                    for (Integer userIdx : (Set<Integer>) itemUsersEntry.getValue()) {
                        userRatingsVector.set(index, trainMatrix.get(userIdx, itemIdx));
                        userPredictsVector.set(index, predictWithoutBound(userIdx, itemIdx));
                        batchUserHiddenMatrix.set(index, userHiddenMatrix.row(userIdx));
                        //double lossError = (trainMatrix.get(userIdx, itemIdx) - predictWithoutBound(userIdx, itemIdx));
                        //loss += lossError * lossError;
                        index++;
                    }

                    for (int factorIdx = 0; factorIdx < hiddenFeatureNum; factorIdx++) {
                        MatrixBasedDenseVector hiddenUsersVector = (MatrixBasedDenseVector) batchUserHiddenMatrix.column(factorIdx);
                        double realRatingValue = hiddenUsersVector.dot(userRatingsVector);
                        double estmRatingValue = hiddenUsersVector.dot(userPredictsVector);
                        double itemHiddenValue = itemHiddenMatrix.get(itemIdx, factorIdx);
                        double error = (realRatingValue - estmRatingValue) - lambdaH * itemHiddenValue;

                        itemHiddenMatrixLearnRate[itemIdx][factorIdx] += error * error;
                        double del = adagrad(itemHiddenMatrixLearnRate[itemIdx][factorIdx], error, itemUsersEntrySize);
                        itemHiddenMatrix.plus(itemIdx, factorIdx, del);
                        if (itemHiddenMatrix.get(itemIdx, factorIdx) < 0)
                            itemHiddenMatrix.set(itemIdx, factorIdx, 0.0);
                        loss += lambdaH * itemHiddenValue * itemHiddenValue;
                    }
                }

                //update featureMatrix
                for (Integer featureIdx : featuresUnion) {
                    Set<Integer> usersSet = new HashSet<>();
                    Set<Integer> itemsSet = new HashSet<>();
                    VectorBasedDenseVector userRatingsVector;
                    VectorBasedDenseVector userPredictsVector;
                    VectorBasedDenseVector itemRatingsVector;
                    VectorBasedDenseVector itemPredictsVector;
                    DenseMatrix batchUserFeatureMatrix, batchItemFeatureMatrix;
                    int featureUsersSize = 0;
                    int featureItemsSize = 0;

                    if (featureUsersSet.containsKey(featureIdx)) {
                        featureUsersSize = featureUsersSet.get(featureIdx).size();
                        userRatingsVector = new VectorBasedDenseVector(featureUsersSize);
                        userPredictsVector = new VectorBasedDenseVector(featureUsersSize);
                        batchUserFeatureMatrix = new DenseMatrix(featureUsersSize, explicitFeatureNum);

                        int index = 0;
                        for (Integer userIdx : featureUsersSet.get(featureIdx)) {
                            userRatingsVector.set(index, userFeatureAttention.get(userIdx, featureIdx));
                            userPredictsVector.set(index, predUserAttention(userIdx, featureIdx));
                            batchUserFeatureMatrix.set(index, userFeatureMatrix.row(userIdx));
                            double lossError = userFeatureAttention.get(userIdx, featureIdx) - predUserAttention(userIdx, featureIdx);
                            loss += lossError * lossError;
                            index++;
                        }
                    } else {
                        userRatingsVector = new VectorBasedDenseVector(0);
                        userPredictsVector = new VectorBasedDenseVector(0);
                        batchUserFeatureMatrix = new DenseMatrix(0, 0);
                    }

                    if (featureItemsSet.containsKey(featureIdx)) {
                        featureItemsSize = featureItemsSet.get(featureIdx).size();
                        itemRatingsVector = new VectorBasedDenseVector(featureItemsSize);
                        itemPredictsVector = new VectorBasedDenseVector(featureItemsSize);
                        batchItemFeatureMatrix = new DenseMatrix(featureItemsSize, explicitFeatureNum);

                        int index = 0;
                        for (Integer itemIdx : featureItemsSet.get(featureIdx)) {
                            itemRatingsVector.set(index, itemFeatureQuality.get(itemIdx, featureIdx));
                            itemPredictsVector.set(index, predItemQuality(itemIdx, featureIdx));
                            batchItemFeatureMatrix.set(index, itemFeatureMatrix.row(itemIdx));
                            double lossError = itemFeatureQuality.get(itemIdx, featureIdx) - predItemQuality(itemIdx, featureIdx);
                            loss += lossError * lossError;
                            index++;
                        }
                    } else {
                        itemRatingsVector = new VectorBasedDenseVector(0);
                        itemPredictsVector = new VectorBasedDenseVector(0);
                        batchItemFeatureMatrix = new DenseMatrix(0, 0);
                    }

                    for (int factorIdx = 0; factorIdx < explicitFeatureNum; factorIdx++) {
                        MatrixBasedDenseVector featureUsersVector = (MatrixBasedDenseVector) batchUserFeatureMatrix.column(factorIdx);
                        MatrixBasedDenseVector featureItemsVector = (MatrixBasedDenseVector) batchItemFeatureMatrix.column(factorIdx);

                        double realUserRatingValue = featureUsersVector.dot(userRatingsVector);
                        double estmUserRatingValue = featureUsersVector.dot(userPredictsVector);
                        double realItemRatingValue = featureItemsVector.dot(itemRatingsVector);
                        double estmItemRatingValue = featureItemsVector.dot(itemPredictsVector);
                        double featureValue = featureMatrix.get(featureIdx, factorIdx);
                        double error = lambdaX * (realUserRatingValue - estmUserRatingValue) + lambdaY * (realItemRatingValue - estmItemRatingValue) - lambdaV * featureValue;
                        featureMatrixLearnRate[featureIdx][factorIdx] += error * error;
                        double del = adagrad(featureMatrixLearnRate[featureIdx][factorIdx], error, featureUsersSize + featureItemsSize);
                        featureMatrix.plus(featureIdx, factorIdx, del);
                        if (featureMatrix.get(featureIdx, factorIdx) < 0.0)
                            featureMatrix.set(featureIdx, factorIdx, 0.0);
                        loss += lambdaV * featureValue * featureValue;
                    }
                }

                //update userFeatureMatrix
                for (Integer userIdx : usersUnion) {
                    VectorBasedDenseVector itemRatingsVector;
                    VectorBasedDenseVector itemPredictsVector;
                    VectorBasedDenseVector featureRatingsVector;
                    VectorBasedDenseVector featurePredictsVector;
                    DenseMatrix batchItemFeatureMatrix, batchFeatureMatrix;
                    int userItemsSize = 0;
                    int userFeaturesSize = 0;

                    if (userItemsSet.containsKey(userIdx)) {
                        userItemsSize = userItemsSet.get(userIdx).size();
                        itemRatingsVector = new VectorBasedDenseVector(userItemsSize);
                        itemPredictsVector = new VectorBasedDenseVector(userItemsSize);
                        batchItemFeatureMatrix = new DenseMatrix(userItemsSize, explicitFeatureNum);

                        int index = 0;
                        for (Integer itemIdx : userItemsSet.get(userIdx)) {
                            itemRatingsVector.set(index, trainMatrix.get(userIdx, itemIdx));
                            itemPredictsVector.set(index, predictWithoutBound(userIdx, itemIdx));
                            batchItemFeatureMatrix.set(index, itemFeatureMatrix.row(itemIdx));
                            double lossError = trainMatrix.get(userIdx, itemIdx) - predictWithoutBound(userIdx, itemIdx);
                            loss += lossError * lossError;
                            index++;
                        }
                    } else {
                        itemRatingsVector = new VectorBasedDenseVector(0);
                        itemPredictsVector = new VectorBasedDenseVector(0);
                        batchItemFeatureMatrix = new DenseMatrix(0, 0);
                    }

                    if (userFeaturesSet.containsKey(userIdx)) {
                        userFeaturesSize = userFeaturesSet.get(userIdx).size();
                        featureRatingsVector = new VectorBasedDenseVector(userFeaturesSize);
                        featurePredictsVector = new VectorBasedDenseVector(userFeaturesSize);
                        batchFeatureMatrix = new DenseMatrix(userFeaturesSize, explicitFeatureNum);

                        int index = 0;
                        for (Integer featureIdx : userFeaturesSet.get(userIdx)) {
                            featureRatingsVector.set(index, userFeatureAttention.get(userIdx, featureIdx));
                            featurePredictsVector.set(index, predUserAttention(userIdx, featureIdx));
                            batchFeatureMatrix.set(index, featureMatrix.row(featureIdx));
                            double lossError = userFeatureAttention.get(userIdx, featureIdx) - predUserAttention(userIdx, featureIdx);
                            loss += lossError * lossError;
                            index++;
                        }
                    } else {
                        featureRatingsVector = new VectorBasedDenseVector(0);
                        featurePredictsVector = new VectorBasedDenseVector(0);
                        batchFeatureMatrix = new DenseMatrix(0, 0);
                    }

                    for (int factorIdx = 0; factorIdx < explicitFeatureNum; factorIdx++) {
                        MatrixBasedDenseVector itemFeatureFactorsVector = (MatrixBasedDenseVector) batchItemFeatureMatrix.column(factorIdx);
                        MatrixBasedDenseVector featureFactorsVector = (MatrixBasedDenseVector) batchFeatureMatrix.column(factorIdx);

                        double realItemRatingValue = itemFeatureFactorsVector.dot(itemRatingsVector);
                        double estmItemRatingValue = itemFeatureFactorsVector.dot(itemPredictsVector);
                        double realFeatureRatingValue = featureFactorsVector.dot(featureRatingsVector);
                        double estmFeatureRatingValue = featureFactorsVector.dot(featurePredictsVector);
                        double userFeatureValue = userFeatureMatrix.get(userIdx, factorIdx);
                        double error = (realItemRatingValue - estmItemRatingValue) + lambdaX * (realFeatureRatingValue - estmFeatureRatingValue) - lambdaU * userFeatureValue;
                        userFeatureMatrixLearnRate[userIdx][factorIdx] += error * error;
                        double del = adagrad(userFeatureMatrixLearnRate[userIdx][factorIdx], error, userItemsSize + userFeaturesSize);
                        userFeatureMatrix.plus(userIdx, factorIdx, del);
                        if (userFeatureMatrix.get(userIdx, factorIdx) < 0.0)
                            userFeatureMatrix.set(userIdx, factorIdx, 0.0);
                    }
                }

                //update itemFeatureMatrix
                for (Integer itemIdx : itemsUnion) {
                    VectorBasedDenseVector userRatingsVector, userPredictsVector;
                    VectorBasedDenseVector featureRatingsVector, featurePredictsVector;
                    DenseMatrix batchUserFeatureMatrix, batchFeatureMatrix;
                    int itemUsersSize = 0;
                    int itemFeaturesSize = 0;

                    if (itemUsersSet.containsKey(itemIdx)) {
                        itemUsersSize = itemUsersSet.get(itemIdx).size();
                        userRatingsVector = new VectorBasedDenseVector(itemUsersSize);
                        userPredictsVector = new VectorBasedDenseVector(itemUsersSize);
                        batchUserFeatureMatrix = new DenseMatrix(itemUsersSize, explicitFeatureNum);

                        int index = 0;
                        for (Integer userIdx : itemUsersSet.get(itemIdx)) {
                            userRatingsVector.set(index, trainMatrix.get(userIdx, itemIdx));
                            userPredictsVector.set(index, predictWithoutBound(userIdx, itemIdx));
                            batchUserFeatureMatrix.set(index, userFeatureMatrix.row(userIdx));
                            double lossError = trainMatrix.get(userIdx, itemIdx) - predictWithoutBound(userIdx, itemIdx);
                            loss += lossError * lossError;
                            index++;
                        }
                    } else {
                        userRatingsVector = new VectorBasedDenseVector(0);
                        userPredictsVector = new VectorBasedDenseVector(0);
                        batchUserFeatureMatrix = new DenseMatrix(0,0);
                    }

                    if (itemFeaturesSet.containsKey(itemIdx)) {
                        itemFeaturesSize = itemFeaturesSet.get(itemIdx).size();
                        featureRatingsVector = new VectorBasedDenseVector(itemFeaturesSize);
                        featurePredictsVector = new VectorBasedDenseVector(itemFeaturesSize);
                        batchFeatureMatrix = new DenseMatrix(itemFeaturesSize, explicitFeatureNum);

                        int index = 0;
                        for (Integer featureIdx : itemFeaturesSet.get(itemIdx)) {
                            featureRatingsVector.set(index, itemFeatureQuality.get(itemIdx, featureIdx));
                            featurePredictsVector.set(index, predItemQuality(itemIdx, featureIdx));
                            batchFeatureMatrix.set(index, featureMatrix.row(featureIdx));
                            double lossError = itemFeatureQuality.get(itemIdx, featureIdx) - predItemQuality(itemIdx, featureIdx);
                            loss += lossError * lossError;
                            index++;
                        }
                    } else {
                        featureRatingsVector = new VectorBasedDenseVector(0);
                        featurePredictsVector = new VectorBasedDenseVector(0);
                        batchFeatureMatrix = new DenseMatrix(0,0);
                    }

                    for (int factorIdx = 0; factorIdx < explicitFeatureNum; factorIdx++) {
                        MatrixBasedDenseVector userFeatureFactorsVector = (MatrixBasedDenseVector) batchUserFeatureMatrix.column(factorIdx);
                        MatrixBasedDenseVector featureFactorsVector = (MatrixBasedDenseVector) batchFeatureMatrix.column(factorIdx);

                        double realUserRatingValue = userFeatureFactorsVector.dot(userRatingsVector);
                        double estmUserRatingValue = userFeatureFactorsVector.dot(userPredictsVector);
                        double realFeatureRatingValue = featureFactorsVector.dot(featureRatingsVector);
                        double estmFeatureRatingValue = featureFactorsVector.dot(featurePredictsVector);
                        double itemFeatureValue = itemFeatureMatrix.get(itemIdx, factorIdx);
                        double error = (realUserRatingValue - estmUserRatingValue) + lambdaY * (realFeatureRatingValue - estmFeatureRatingValue) - lambdaU * itemFeatureValue;
                        itemFeatureMatrixLearnRate[itemIdx][factorIdx] += error * error;
                        double del = adagrad(itemFeatureMatrixLearnRate[itemIdx][factorIdx], error, itemUsersSize + itemFeaturesSize);
                        itemFeatureMatrix.plus(itemIdx, factorIdx, del);
                        if (itemFeatureMatrix.get(itemIdx, factorIdx) < 0.0)
                            itemFeatureMatrix.set(itemIdx, factorIdx, 0.0);
                    }

                }
            }
            LOG.info("iter:" + iter + ", loss:" + loss);
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
        List<Map.Entry<K, V>> list = new LinkedList<>( map.entrySet() );
        Collections.sort(list, new Comparator<Map.Entry<K, V>>()
        {
            public int compare( Map.Entry<K, V> o1, Map.Entry<K, V> o2 )
            {
                return (o1.getValue()).compareTo( o2.getValue() );
            }
        } );

        Map<K, V> result = new LinkedHashMap<K, V>();
        for (Map.Entry<K, V> entry : list)
        {
            result.put( entry.getKey(), entry.getValue() );
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
        return  (rating - minRate) / (maxRate - minRate);
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

    protected static class Differential {
        Map<Integer, Set<Integer>> batchSet;
        VectorBasedDenseVector ratingsVector;
        VectorBasedDenseVector predictsVector;
        DenseMatrix batchMatrix;
        int setSize;

        public Differential(Map<Integer, Set<Integer>> batchSet) {
            this.batchSet = batchSet;
        }

        public void calcSetUp(int keyIdx,  int numFactors) {
            if (batchSet.containsKey(keyIdx)) {
                setSize = batchSet.get(keyIdx).size();
                ratingsVector = new VectorBasedDenseVector(setSize);
                predictsVector = new VectorBasedDenseVector(setSize);
                batchMatrix = new DenseMatrix(setSize, numFactors);
            } else {
                ratingsVector = new VectorBasedDenseVector(0);
                predictsVector = new VectorBasedDenseVector(0);
                batchMatrix = new DenseMatrix(0,0);
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

        double realRatingValue(Vector dotVector) {
            return ratingsVector.dot(dotVector);
        }

        double estmRatingValue(Vector dotVector) {
            return predictsVector.dot(dotVector);
        }
    }

    protected double topKPredict(int u, int j) {
        double tradeoff = conf.getDouble("rec.tradeoff", 0.0);
        double pred = (1.0 - tradeoff) * predictWithoutBound(u, j);

        double[] userFeatureValues = new double[numberOfFeatures];
        double[] recItemFeatureValues = new double[numberOfFeatures];
        /**
         if (userFeatureValuesFlag[u] == false) {
         userFeatureValuesFlag[u] = true;
         userFeatureValues = featureMatrix.times(userFeatureMatrix.row(u)).getValues();
         userFeatureValuesMatrix.set(u, new VectorBasedDenseVector(userFeatureValues));
         } else {
         userFeatureValues = userFeatureValuesMatrix.row(u).getValues();
         }

         if (recItemFeatureValuesFlag[j] == false) {
         recItemFeatureValuesFlag[j] = true;
         recItemFeatureValues = featureMatrix.times(itemFeatureMatrix.row(j)).getValues();
         recItemFeatureValuesMatrix.set(j, new VectorBasedDenseVector(recItemFeatureValues));
         } else {
         recItemFeatureValues = recItemFeatureValuesMatrix.row(j).getValues();
         }
         **/
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
        for (int i=0; i<numFeatureTopK; i++) {
            int featureIdx = (int) userTopFeatureIndices[numFeatureTopK - 1 - i];
            userTopFeatureValues[i] = userFeatureValues[featureIdx];
            recItemTopFeatureValues[i] = recItemFeatureValues[featureIdx];
        }

        //calc k largest values
        double largestValue = 0.0;
        for (int i = 0; i < numFeatureTopK; i++) {
            largestValue += userTopFeatureValues[i] * recItemFeatureValues[i];
        }
        largestValue *= (tradeoff / (numFeatureTopK * maxRate));
        pred += largestValue;

        return pred;
    }


}
