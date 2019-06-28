package recommend;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import org.apache.commons.lang.StringUtils;

import com.google.common.collect.BiMap;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.HashBiMap;
import com.google.common.collect.Table;

import net.librec.common.LibrecException;
import net.librec.math.structure.DataFrame;
import net.librec.math.structure.DenseMatrix;
import net.librec.math.structure.DenseVector;
import net.librec.math.structure.MatrixEntry;
import net.librec.math.structure.RandomAccessSparseVector;
import net.librec.math.structure.SequentialAccessSparseMatrix;
import net.librec.math.structure.SequentialSparseVector;
import net.librec.math.structure.TensorEntry;
import net.librec.math.structure.VectorBasedDenseVector;
import net.librec.recommender.TensorRecommender;

/**
 * EFM Recommender
 * Zhang Y, Lai G, Zhang M, et al. Explicit factor models for explainable recommendation based on phrase-level sentiment analysis[C]
 * {@code Proceedings of the 37th international ACM SIGIR conference on Research & development in information retrieval.  ACM, 2014: 83-92}.
 *
 * @author ChenXu and SunYatong
 */
public class EfmRecommender extends TensorRecommender {

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
    protected BiMap<String, Integer> featureDict;

    protected SequentialAccessSparseMatrix trainMatrix;

    public BiMap<Integer, String> featureSentimemtPairsMappingData;
    public BiMap<Integer, String> userPairsMappingData;

    protected ArrayList<Map<Integer, Double>> userFeatureValuesList;
    protected DenseMatrix userFeatureValuesMatrix;
    protected DenseMatrix recItemFeatureValuesMatrix;
    protected boolean[] userFeatureValuesFlag;
    protected boolean[] recItemFeatureValuesFlag;
    protected ArrayList<VectorBasedDenseVector> userFeatureValuesVectorList;
    protected ArrayList<VectorBasedDenseVector> recItemFeatureValuesVectorList;

    boolean doExplain;
    boolean doRanking;
    boolean doNormal;
    boolean doTfIdf;
    boolean doSim;


    /*
     * (non-Javadoc)
     *
     * @see net.librec.recommender.AbstractRecommender#setup()
     */
    @Override
    protected void setup() throws LibrecException {
        super.setup();
        //scoreScale = maxRate - minRate;
        scoreScale = maxRate;
        explicitFeatureNum = conf.getInt("rec.factor.explicit", 5);
        hiddenFeatureNum = numFactors - explicitFeatureNum;
        lambdaX = conf.getDouble("rec.regularization.lambdax", 0.001);
        lambdaY = conf.getDouble("rec.regularization.lambday", 0.001);
        lambdaU = conf.getDouble("rec.regularization.lambdau", 0.001);
        lambdaH = conf.getDouble("rec.regularization.lambdah", 0.001);
        lambdaV = conf.getDouble("rec.regularization.lambdav", 0.001);


        doNormal = conf.getBoolean("rec.weight.normal", false);
        doTfIdf= conf.getBoolean("rec.weight.tfIdf", false);
        doSim = conf.getBoolean("rec.weight.sim", false);

        featureSentimemtPairsMappingData = DataFrame.getInnerMapping("sentiment").inverse();
        trainMatrix = trainTensor.rateMatrix();

        featureDict = HashBiMap.create();
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
        Map<Integer, String> userFeatureDict = new HashMap<Integer, String>();
        Map<Integer, String> itemFeatureDict = new HashMap<Integer, String>();

        numberOfFeatures = 0;

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


        Table<Integer, String, String> helpUserFeatureTable = HashBasedTable.create();
        Table<Integer, Integer, Double> helpUserSimilarityTable = HashBasedTable.create();
        if (doNormal && doSim) {
            addHelpDict(helpUserFeatureTable, helpUserSimilarityTable);
        }
        if (doNormal && !doTfIdf && !doSim) {
            addSentiment(userFeatureDict, itemFeatureDict);
        }
        if (!doNormal && doTfIdf) {
            tfIdfWeight(userFeatureDict, itemFeatureDict);
        }

        // Create V,U1,H1,U2,H2
        featureMatrix = new DenseMatrix(numberOfFeatures, explicitFeatureNum);
        featureMatrix.init(0.01);
        userFeatureMatrix = new DenseMatrix(numUsers, explicitFeatureNum); //userFactors.getSubMatrix(0, userFactors.numRows() - 1, 0, explicitFeatureNum - 1);
        userFeatureMatrix.init(1);
        userHiddenMatrix = new DenseMatrix(numUsers, numFactors - explicitFeatureNum); // userFactors.getSubMatrix(0, userFactors.numRows() - 1, explicitFeatureNum, userFactors.numColumns() - 1);
        userHiddenMatrix.init(1);
        itemFeatureMatrix = new DenseMatrix(numItems, explicitFeatureNum);// itemFactors.getSubMatrix(0, itemFactors.numRows() - 1, 0, explicitFeatureNum - 1);
        itemFeatureMatrix.init(1);
        itemHiddenMatrix = new DenseMatrix(numItems, numFactors - explicitFeatureNum);// itemFactors.getSubMatrix(0, itemFactors.numRows() - 1, explicitFeatureNum, itemFactors.numColumns() - 1);
        itemHiddenMatrix.init(1);

        // compute UserFeatureAttentioncxyygty
        Table<Integer, Integer, Double> userFeatureAttentionTable = HashBasedTable.create();
        double userSlope = conf.getDouble("rec.efm.user.slope", 1.0);
        //double userBias = conf.getDouble("rec.efm.user.bias", 0.0);
        double userBias = 0.0;
        for (int u : userFeatureDict.keySet()) {
            double[] featureValues = new double[numberOfFeatures];
            String[] fList = userFeatureDict.get(u).split(" ");
            for (String a : fList) {
                if (!StringUtils.isEmpty(a)) {
                    int fin = featureDict.get(a.split(":")[0]);
                    if (!doTfIdf) {
                        featureValues[fin] += 1;
                    } else {
                        featureValues[fin] += Math.abs(Double.parseDouble(a.split(":")[1]));
                    }
                }
            }
            for (int i = 0; i < numberOfFeatures; i++) {
                if (featureValues[i] != 0.0) {
                    if (doTfIdf || doNormal) {
                        double lowerLimitUserFeatureValues = conf.getDouble("rec.weight.lowerLimit", 0.0);
                        if (featureValues[i] > lowerLimitUserFeatureValues) {
                            double v = 1 + (scoreScale - 1) * (2 / (1 + Math.exp(linearTrans(userSlope, userBias, -featureValues[i]))) - 1);
                            userFeatureAttentionTable.put(u, i, v);
                        }
                    } else if (!doTfIdf || !doNormal){
                        double v = 1 + (scoreScale - 1) * (2 / (1 + Math.exp(linearTrans(userSlope, userBias, -featureValues[i]))) - 1);
                        userFeatureAttentionTable.put(u, i, v);
                    }
                }
            }
        }
        updateFeatureTableBySimilarity(userFeatureAttentionTable, helpUserFeatureTable, helpUserSimilarityTable, "user");
        userFeatureAttention = new SequentialAccessSparseMatrix(numUsers, numberOfFeatures, userFeatureAttentionTable);

        // Compute ItemFeatureQuality
        Table<Integer, Integer, Double> itemFeatureQualityTable = HashBasedTable.create();
        double itemSlope = conf.getDouble("rec.efm.item.slope", 1.0);
        double itemBias = conf.getDouble("rec.efm.item.bias", 0.0);
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
                    if (doTfIdf) {
                        //todo need to modify rec.weight.LowerLimit to rec.weight.itemLowerLimit
                        double lowerLimitItemFeatureValues = conf.getDouble("rec.weight.lowerLimit", 0.0);
                        if (featureValues[i] > lowerLimitItemFeatureValues || featureValues[i] < -lowerLimitItemFeatureValues) {
                            double v = 1 + (scoreScale - 1) / (1 + Math.exp(linearTrans(itemSlope, itemBias, -featureValues[i])));
                            itemFeatureQualityTable.put(p, i, v);
                        }
                    } else if (!doTfIdf){
                        double v = 1 + (scoreScale - 1) / (1 + Math.exp(linearTrans(itemSlope, itemBias, -featureValues[i])));
                        itemFeatureQualityTable.put(p, i, v);
                    }
                }
            }
        }
        itemFeatureQuality = new SequentialAccessSparseMatrix(numItems, numberOfFeatures, itemFeatureQualityTable);

        doExplain = conf.getBoolean("rec.explain.flag");
        LOG.info("numUsers:" + numUsers);
        LOG.info("numItems:" + numItems);
        LOG.info("numFeatures:" + numberOfFeatures);
    }

    @Override
    protected void trainModel() throws LibrecException {
        for (int iter = 1; iter <= conf.getInt("rec.iterator.maximum"); iter++) {
            loss = 0.0;
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

    protected void addHelpDict(Table helpFeatureTable, Table helpSimilarityTable) throws IllegalAccessError{
        BiMap<Integer, String> userHelpPairsMappingData = DataFrame.getInnerMapping("help_user").inverse();
        BiMap<Integer, String> featureSentimentHelpPairsMappingData = DataFrame.getInnerMapping("help_sentiment").inverse();
        BiMap<Integer, String> featureSentimentSimPairsMappingData = DataFrame.getInnerMapping("help_sim").inverse();
        BiMap<Integer, String> userPairsMappingData = DataFrame.getInnerMapping("user").inverse();

        if (userHelpPairsMappingData == null || featureSentimentHelpPairsMappingData == null) {
            throw new IllegalAccessError("userHelpPairsMappingData or featureSentimentHelpPairsMappingData is null");
        }

        for (TensorEntry te : trainTensor) {
            int[] entryKeys = te.keys();
            int userIndex = entryKeys[0];
            int itemIndex = entryKeys[1];
            //note rating don't include entryKeys so the number of entryKeys is |Attribute| - 1
            int userHelpPairsIndex = entryKeys[3];
            int featureSentimentHelpPairsIndex = entryKeys[5];

            String userHelpPairsString = userHelpPairsMappingData.get(userHelpPairsIndex)
                    .replaceAll("</endperson[0-9]+>", "");
            /**
             * uHPList = [helpUserIdx1, helpUserIdxk...]
             */
            String[] uHPList = userHelpPairsString.split(";;;;");

            String featureSentimentHelpPairsString = featureSentimentHelpPairsMappingData.get(featureSentimentHelpPairsIndex)
                    .replaceAll("</endperson[0-9]+>", "");
            /**
             * userFSHPList [sentimentString1,setimentString2,...]
             */
            String[] userFSHPList = featureSentimentHelpPairsString.split(";;;;");
            int idx = 0;
            for (String helpUserIdxStr : uHPList) {
                String[] helpUserSentimentsList = userFSHPList[idx].split(" ");

                for (String helpUserSentiment : helpUserSentimentsList) {
                    String sentimentWord = helpUserSentiment.split(":")[0];

                    if (!helpFeatureTable.contains(userIndex, sentimentWord) && !StringUtils.isEmpty(sentimentWord)) {
                        String sentimentValue = helpUserSentiment.split(":")[1];
                        String idxAndValue = helpUserIdxStr + ":" + sentimentValue;
                        helpFeatureTable.put(userIndex, sentimentWord, idxAndValue);
                    } else if (!StringUtils.isEmpty(sentimentWord)){
                        String sentimentValue = helpUserSentiment.split(":")[1];
                        String idxAndValue = helpUserIdxStr + ":" + sentimentValue;
                        String addToIdxAndValue = helpFeatureTable.get(userIndex, sentimentWord) + " " + idxAndValue;
                        helpFeatureTable.put(userIndex, sentimentWord, addToIdxAndValue);
                    }
                }

                if (featureSentimentHelpPairsMappingData != null && !StringUtils.isEmpty(helpUserIdxStr)) {
                    int firstIdx, secondIdx;
                    if (userIndex <= Integer.valueOf(helpUserIdxStr)) {
                        firstIdx = userIndex;
                        secondIdx = Integer.valueOf(helpUserIdxStr);
                    } else {
                        firstIdx = Integer.valueOf(helpUserIdxStr);
                        secondIdx = userIndex;
                    }

                    if (!helpSimilarityTable.contains(firstIdx, secondIdx)) {
                        helpSimilarityTable.put(firstIdx, secondIdx, simWeight(te, idx, featureSentimentSimPairsMappingData));
                    }
                }
                idx++;
            }
        }
    }

    protected void updateFeatureTableBySimilarity(Table<Integer, Integer, Double> featureTable, Table<Integer, String, String> helpFeatureTable ,
                                                  Table<Integer, Integer, Double> similarityTable, String userOrItemName) throws LibrecException{
        double coefficientBeta = conf.getDouble("rec.weight.beta", 0.7);
        for (Table.Cell<Integer, String, String> cell : helpFeatureTable.cellSet()) {
            int helpedIdx = cell.getRowKey();
            String sentimentWord = cell.getColumnKey();
            String[] helpIdxAndValueList = cell.getValue().split(" ");

            if (!featureDict.containsKey(sentimentWord))
                continue;

            int featureIdx = featureDict.get(sentimentWord);
            double featureFrequency = 0.0;
            double weightSlope = 0.0;
            double numberOfHelpFeature = helpIdxAndValueList.length;
            for (String helpIdxAndValue : helpIdxAndValueList) {
                int helpIdx = Integer.valueOf(helpIdxAndValue.split(":")[0]);
                double sentimentValue = Double.valueOf(helpIdxAndValue.split(":")[1]);

                int firstIdx, secondIdx;
                if (helpedIdx <= helpIdx) {
                    firstIdx = helpedIdx;
                    secondIdx = helpIdx;
                } else {
                    firstIdx = helpIdx;
                    secondIdx = helpedIdx;
                }

                double simValue = similarityTable.get(firstIdx, secondIdx);
                if (userOrItemName.equals("user")) {
                    if (simValue >= 0.15) {
                        weightSlope += simValue * Math.abs(sentimentValue) / numberOfHelpFeature;
                        featureFrequency += Math.abs(sentimentValue);
                    }
                }
                else if (userOrItemName.equals("item")) {
                    weightSlope += simValue * sentimentValue / numberOfHelpFeature;
                    featureFrequency += sentimentValue;
                }
            }
            if (featureFrequency != 0.0) {
                double bias = conf.getDouble("rec.efm." + userOrItemName + "." + "bias");
                double slopeCoefficient = conf.getDouble("rec.weight.slopeCoefficient", 1.0);
                double updateFeatureValue;
                if (userOrItemName.equals("user")) {
                    updateFeatureValue = getUserFeatureAttentionValue(slopeCoefficient * weightSlope, bias, featureFrequency);
                } else if (userOrItemName.equals("item")) {
                    updateFeatureValue = getItemFeatureQualityValue(slopeCoefficient * weightSlope, bias, featureFrequency);
                } else {
                    throw new LibrecException("invalid conf bias name" + userOrItemName);

                }

                if (featureTable.contains(helpedIdx, featureIdx)) {
                    featureTable.put(helpedIdx, featureIdx,
                            coefficientBeta * featureTable.get(helpedIdx, featureIdx) + (1.0 - coefficientBeta) * updateFeatureValue
                    );
                } else {
                    //if ((0.0 < updateFeatureValue && updateFeatureValue < 1.7) || updateFeatureValue > 4.0)
                    //if ( updateFeatureValue > 3.0)
                        featureTable.put(helpedIdx, featureIdx, updateFeatureValue);
                    //featureTable.put(helpedIdx, featureIdx, 3.0);
                }
            }
        }
    }

    protected double getUserFeatureAttentionValue(double userSlope, double userBias, double userFeatureFrequency) {
        //todo
        //return 1 + (scoreScale - 1) * (2 / (1 + Math.exp(linearTrans(userSlope, userBias, -userFeatureFrequency))) - 1);
        //fuzzy gaussian membership function(experience)
        //return 1 + (scoreScale - 1) *(Math.exp(-Math.pow((8 - userFeatureFrequency), 2) / 2 * Math.pow((1 - userSlope), 2)));
        //experience
        return 1 + (scoreScale - 1) * (2 / (1 + Math.exp(linearTrans(-6.0 , 0.0, userSlope))) - 1);
        //return 1 + (scoreScale - 1) * (2 / (1 + Math.exp(linearTrans(0.9, 0.5, -userFeatureFrequency))) - 1);
    }

    protected double getItemFeatureQualityValue(double itemSlope, double itemBias, double itemFeatureFrequency) {
        //todo
        return 1.0;
    }

    protected void addSentiment(Map userFeatureDict, Map itemFeatureDict) {
        //featureSentimemtPairsMappingData = DataFrame.getInnerMapping("sentiment").inverse();
        BiMap<Integer, String> userHelpPairsMappingData = DataFrame.getInnerMapping("help_user").inverse();
        BiMap<Integer, String> featureSentimentHelpPairsMappingData = DataFrame.getInnerMapping("help_sentiment").inverse();
        //BiMap<Integer, String> userPairsMappingData = DataFrame.getInnerMapping("user").inverse();

        for (TensorEntry te : trainTensor) {
            int[] entryKeys = te.keys();
            int userIndex = entryKeys[0];
            int itemIndex = entryKeys[1];
            int userHelpPairsIndex = entryKeys[3];
            int featureSentimentHelpPairsIndex = entryKeys[5];

            String userHelpPairsString = userHelpPairsMappingData.get(userHelpPairsIndex)
                    .replaceAll("</endperson[0-9]+>", "");
            String[] uHPList = userHelpPairsString.split(";;;;");

            String featureSentimentHelpPairsString = featureSentimentHelpPairsMappingData.get(featureSentimentHelpPairsIndex)
                    .replaceAll("</endperson[0-9]+>", "");
            /**
             * userFSHPList [sentimentString1,setimentString2,...]
             */
            String[] userFSHPList = featureSentimentHelpPairsString.split(";;;;");

            int idx = 0;
            for (String userFSHP: userFSHPList) {
                if (uHPList.length == 0 || uHPList[idx].isEmpty())
                    continue;
                int userHelpIndex = Integer.parseInt(uHPList[idx]);
                //userHelpIndex = userPairsMappingData.inverse().get(uHPList[idx]);

                //add user help data if not in test data
                if (testTensor.getIndices(userHelpIndex, itemIndex).isEmpty()) {
                    String[] uHList = userFSHP.split(" ");
                    for (String p : uHList) {
                        if (p.isEmpty())
                            continue;
                        else {
                            String updateSentiment = "";
                            String sentimentWord = p.split(":")[0];
                            double sentimentValue = Double.valueOf(p.split(":")[1]);
                            double coefficient = conf.getDouble("rec.weight.coefficient", 1.0);
                            updateSentiment += sentimentWord + ":" + String.valueOf(
                                         coefficient * sentimentValue
                            );
                            if (!userFeatureDict.containsKey(userIndex)) {
                                userFeatureDict.put(userIndex, updateSentiment);
                            } else {
                                userFeatureDict.put(userIndex, userFeatureDict.get(userIndex) + " " + updateSentiment);
                            }
                            if (!itemFeatureDict.containsKey(itemIndex))
                                itemFeatureDict.put(itemIndex, updateSentiment);
                            else
                                itemFeatureDict.put(itemIndex, itemFeatureDict.get(itemIndex) + " " + updateSentiment);
                        }
                    }
                }
                idx++;
            }
        }

    }


    protected void tfIdfWeight(Map userFeatureDict, Map itemFeatureDict) {
        BiMap<Integer, String> userHelpPairsMappingData = DataFrame.getInnerMapping("help_user").inverse();
        BiMap<Integer, String> featureSentimentHelpPairsMappingData = DataFrame.getInnerMapping("help_sentiment").inverse();
        BiMap<Integer, String> featureSentimentTfIdfPairsMappingData = DataFrame.getInnerMapping("help_tfIdf").inverse();
        BiMap<Integer, String> featureSentimentSimPairsMappingData = HashBiMap.create();

        if (doSim) {
            int entrySimIdx = conf.getInt("rec.weight.sim.entryIdx", 7);
            featureSentimentSimPairsMappingData = DataFrame.getInnerMapping("help_sim").inverse();
        }


        double cofficient = conf.getDouble("rec.weight.coefficient", 1.0);


        for (TensorEntry te : trainTensor) {
            int[] entryKeys = te.keys();
            int userIndex = entryKeys[0];
            int itemIndex = entryKeys[1];
            int userHelpPairsIndex = entryKeys[3];
            int featureSentimentHelpPairsIndex = entryKeys[5];
            int featureSentimentTfIdfPairsIndex = entryKeys[6];

            String userHelpPairsString = userHelpPairsMappingData.get(userHelpPairsIndex)
                    .replaceAll("</endperson[0-9]+>", "");
            String[] uHPList = userHelpPairsString.split(";;;;");
            String featureSentimentHelpPairsString = featureSentimentHelpPairsMappingData.get(featureSentimentHelpPairsIndex)
                    .replaceAll("</endperson[0-9]+>", "");
            String featureSentimentTfIdfPairsString = featureSentimentTfIdfPairsMappingData.get(featureSentimentTfIdfPairsIndex)
                    .replaceAll("</endperson[0-9]+>", "");
            /**
             * userFSHPList [sentimentString1,setimentString2,...]
             */
            String[] userFSHPList = featureSentimentHelpPairsString.split(";;;;");
            String[] userFSTfIdfPList = featureSentimentTfIdfPairsString.split(";;;;");

            for (int helpUserIdx = 0; helpUserIdx < userFSHPList.length; helpUserIdx++) {
                String userFSHP = userFSHPList[helpUserIdx];
                String userFSTfIdfP = userFSTfIdfPList[helpUserIdx];

                if (uHPList.length == 0 || uHPList[helpUserIdx].isEmpty())
                    continue;
                int userHelpIndex = Integer.parseInt(uHPList[helpUserIdx]);

                if (testTensor.getIndices(userHelpIndex, itemIndex).isEmpty()) {
                    //split sentiment1:value1, sentiment2:value2
                    //split sentiment1:[value1|value1'...], sentiment2:
                    String[] uFSHP = userFSHP.split(" ");
                    String[] uFSTfIdfP = userFSTfIdfP.split(" ");


                    for (int wordIdx = 0; wordIdx < uFSHP.length; wordIdx++) {

                        if (uFSHP[wordIdx].isEmpty())
                            continue;
                        String featureName = uFSHP[wordIdx].split(":")[0];
                        Double featureValue = Double.valueOf(uFSHP[wordIdx].split(":")[1]);

                        //tfIdf
                        String featureTIPairs = uFSTfIdfP[wordIdx].replaceAll("\\[", "")
                                .replaceAll("\\]", "");
                        String[] featureValueTIStringList = featureTIPairs.split(":")[1].split("\\|");
                        double  maxTIValue = 0.0;

                        for (String fVTIS : featureValueTIStringList)
                            maxTIValue = Math.max(maxTIValue, Double.valueOf(fVTIS));

                        String p = featureName + ":" + String.valueOf(
                                cofficient * featureValue * maxTIValue * simWeight(te, helpUserIdx, featureSentimentSimPairsMappingData)
                        );

                        if (userFeatureDict.containsKey(userIndex))
                            userFeatureDict.put(userIndex, userFeatureDict.get(userIndex) + " " + p);
                        else
                            userFeatureDict.put(userIndex, p);

                        if (itemFeatureDict.containsKey(itemIndex))
                            itemFeatureDict.put(itemIndex, itemFeatureDict.get(itemIndex) + " " + p);
                        else
                            itemFeatureDict.put(itemIndex, p);
                    }
                }
            }
        }

    }


    protected double simWeight(TensorEntry te, int listIdx, BiMap<Integer, String> featureSentimentSimParisMappingData) throws IllegalAccessError{
        if (doSim && featureSentimentSimParisMappingData == null) {
            throw new IllegalAccessError("featureSentimentSimPairsMappingData is null");
        }
        if (!doSim)
            return 1.0;

        int entryIdx = conf.getInt("rec.weight.sim.entryIdx", 7);
        int[] entryKeys = te.keys();
        int featureSentimentSimIndex = entryKeys[entryIdx];

        String featureSentimentHelpSimPairsString = featureSentimentSimParisMappingData.get(featureSentimentSimIndex)
                .replaceAll("</endperson[0-9]+>", "");
        String[] fSHSP = featureSentimentHelpSimPairsString.split(";;;;");

        return Double.valueOf(fSHSP[listIdx]);

    }

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

    protected double linearTrans(double slope, double bias, double x) {
        return slope * x + bias;
    }

    protected double sigmoid(double gain, double bias, double x) { return 1.0 / (1.0 + Math.exp(-gain * x + bias)); }

    //protected double fuzzyGaussian

    @Override
    protected double predict(int[] indices) {
        return predict(indices[0], indices[1]);
    }

    protected double predict(int u, int j) {
        doRanking = conf.getBoolean("rec.recommend.doRanking", false);
        if (doRanking && isRanking) {
            double pred = topKPredict(u, j);
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
}

