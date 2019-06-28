package recommend;

import com.google.common.collect.BiMap;
import net.librec.math.structure.SequentialAccessSparseMatrix;
import batch.BatchSet;
import com.google.common.collect.*;
import net.librec.common.LibrecException;
import net.librec.conf.Configuration;
import net.librec.math.algorithm.Maths;
import net.librec.math.algorithm.Randoms;
import net.librec.math.structure.*;
import org.apache.commons.lang.StringUtils;

import java.util.*;

public class EfmSGDRecommender extends EfmRecommender{
    protected SequentialAccessSparseMatrix userHelpfulFeatureAttention;
    protected SequentialAccessSparseMatrix sideRatingMatrix;
    protected BiMap<Integer, String> sideRatingPairsMappingData;
    protected Table<Integer, Integer, String> userItemFeaturesTable;
    protected Table<Integer, Integer, String> userItemHelpFeaturesTable;
    protected double epsilon;
    protected double eta;
    protected int batchSize;

    protected BiMap<Integer, String> userDict;
    protected Map<Integer, String> userHelpedDict;


    @Override
    protected void setup() throws LibrecException {
        super.setup();
        //featureMatrix.init(0.1);
        sideRatingPairsMappingData = DataFrame.getInnerMapping("side").inverse();
        BiMap<String, Integer> userEncodePairsMappingData = DataFrame.getInnerMapping("user");
        Table<Integer, Integer, Double> sideRatingTable = HashBasedTable.create();
        Table<Integer, Integer, Double> userHelpfulFeatureAttentionTable = HashBasedTable.create();
        userHelpedDict = new HashMap<>();
        userItemFeaturesTable = HashBasedTable.create();
        for (TensorEntry te : trainTensor) {
            int[] entryKeys = te.keys();
            int userIndex = entryKeys[0];
            int itemIndex = entryKeys[1];
            int featureSentimentPairsIndex = entryKeys[2];
            int sideRatingPairsIndex = entryKeys[3];

            String sideRatingPairsString = sideRatingPairsMappingData.get(sideRatingPairsIndex)
                    .replaceAll("</endperson[0-9]+>", "");
            String featureSentimentPairsString = featureSentimemtPairsMappingData.get(featureSentimentPairsIndex)
                    .replaceAll("</endperson[0-9]+>", "");
            if (!sideRatingPairsString.isEmpty() && !featureSentimentPairsString.isEmpty()) {
                String[] fSPList = featureSentimentPairsString.split(" ");
                for (String featureAndSentiment : fSPList) {
                    String feature = featureAndSentiment.split(":")[0];
                    String[] sRPList = sideRatingPairsString.split(" ");

                    for (String srp : sRPList) {
                        String helpedUserIdxString = srp.split(";")[0];
                        int helpedUserEncodeIdx = Integer.valueOf(userEncodePairsMappingData.get(helpedUserIdxString));
                        int featureIdx = featureDict.get(feature);
                        double similarity = Double.valueOf(srp.split(";")[3]);
                        userHelpfulFeatureAttentionTable.put(helpedUserEncodeIdx, featureIdx, similarity);
                    }
                }
            }

            //userDict
            //if (!userDict.containsKey(userIndex))
            //    userDict.put(userIndex, String.valueOf(itemIndex) + ";" + featureSentimentPairsString);
            //else
            //    userDict.put(userIndex, userDict.get(userIndex) + " " + String.valueOf(itemIndex) + ";" + featureSentimentPairsString);

            // userHelpedDict
            String[] sideRatingPairsList = sideRatingPairsString.split(" ");
            for (String srp : sideRatingPairsList) {
                if (srp.isEmpty())
                    continue;
                String userHelped = srp.split(";")[0];
                int userHelpedEncode = Integer.valueOf(userEncodePairsMappingData.get(userHelped));
                if (!userHelpedDict.containsKey(userHelpedEncode))
                    userHelpedDict.put(userHelpedEncode, String.valueOf(userIndex) + ":" + String.valueOf(itemIndex));
                else
                    userHelpedDict.put(userHelpedEncode, userHelpedDict.get(userHelpedEncode) + " " + String.valueOf(userIndex) + ":" + String.valueOf(itemIndex));
            }
            if (!featureSentimentPairsString.isEmpty())
                userItemFeaturesTable.put(userIndex, itemIndex, featureSentimentPairsString);
        }

        userHelpfulFeatureAttention = new SequentialAccessSparseMatrix(numUsers, numberOfFeatures, userHelpfulFeatureAttentionTable);
    }


    protected void trainModel() throws LibrecException {
        //adagrad(sgd)
        epsilon = conf.getDouble("rec.sgd.epsilon", 1e-8);
        eta = conf.getDouble("rec.sgd.eta", 1.0);
        batchSize = conf.getInt("rec.sgd.batchSize", 300);
        double[][] featureMatrixLearnRate = new double[numberOfFeatures][explicitFeatureNum];
        double[][] userFeatureMatrixLearnRate = new double[numUsers][explicitFeatureNum];
        double[][] userHiddenMatrixLearnRate = new double[numUsers][numFactors - explicitFeatureNum];
        double[][] itemFeatureMatrixLearnRate = new double[numItems][explicitFeatureNum];
        double[][] itemHiddenMatrixLearnRate = new double[numItems][numFactors - explicitFeatureNum];
        List<Set<Integer>> userItemsSet = getRowColumnsSet(trainMatrix, numUsers);
        List<Set<Integer>> userFeaturesSet = getRowColumnsSet(userFeatureAttention, numUsers);
        List<Set<Integer>> itemFeaturesSet = getRowColumnsSet(itemFeatureQuality, numItems);
        List<Set<Integer>> userHelpFeaturesSet = getRowColumnsSet(userHelpfulFeatureAttention, numUsers);
        List<Set<Integer>> userFeaturesUnion = getUnion(userFeaturesSet, userHelpFeaturesSet);

        List<Set<Integer>> userFeatureUnionSet = new ArrayList<>();
        for (int userIdx = 0; userIdx < numUsers; userIdx++) {
            Set<Integer> union = new HashSet<>(userFeaturesSet.get(userIdx));
            union.addAll(userHelpFeaturesSet.get(userIdx));
            userFeatureUnionSet.add(union);
        }

        for (int iter = 1; iter <= numIterations; iter++) {
            int maxSample = (trainMatrix.size() + userFeatureAttention.size() + itemFeatureQuality.size() + userHelpfulFeatureAttention.size()) / (4 * batchSize);

            loss = 0.0d;
            for (int sampleCount = 0; sampleCount < maxSample; sampleCount++) {
                ArrayList<Map<String, Integer>> sampleSet = new ArrayList<>();
                Map<String, Multimap<Integer, Integer>> sampleIndices = new HashMap<>();
                sampleIndices.put("user", HashMultimap.create());
                sampleIndices.put("item", HashMultimap.create());
                sampleIndices.put("feature", HashMultimap.create());

                Set<Integer> itemUserPosFeatureUnion = new HashSet<>();
                int sampleSize = 0;
                while (sampleSize < batchSize) {
                    Map<String, Integer> sampleMap = new HashMap<>();
                    Multimap<Integer, Integer> userSample = sampleIndices.get("user");
                    Multimap<Integer, Integer> itemSample = sampleIndices.get("item");
                    Multimap<Integer, Integer> featureSample = sampleIndices.get("feature");
                    int userIdx = Randoms.uniform(numUsers);
                    Set<Integer> itemsSet = userItemsSet.get(userIdx);
                    if (itemsSet.size() == 0 || itemsSet.size() == numItems)
                        continue;
                    int[] itemIndices = trainMatrix.row(userIdx).getIndices();
                    int itemIdx = itemIndices[Randoms.uniform(itemIndices.length)];

                    //user feature, negative feature sampling
                    int featureIdx = -1;
                    if (userItemFeaturesTable.contains(userIdx, itemIdx)) {
                        Set<Integer> uFUnion = userFeaturesUnion.get(userIdx);
                        String[] featuresList = userItemFeaturesTable.get(userIdx, itemIdx).split(" ");
                        String featureString = featuresList[Randoms.uniform(featuresList.length)].split(":")[0];
                        featureIdx = featureDict.get(featureString);
                    }
                    /*
                    while (true) {
                        Set<Integer> uFUnion = userFeaturesUnion.get(userIdx);
                        Set<Integer> uFSet = userFeaturesSet.get(userIdx);

                        if (uFSet.size() == 0 || uFSet.size() == numberOfFeatures)
                            break;
                        int[] featureIndices = userFeatureAttention.row(userIdx).getIndices();
                        featureIdx = featureIndices[Randoms.uniform(featureIndices.length)];
                        do {
                            negFeatureIdx = Randoms.uniform(numberOfFeatures);
                        } while (uFUnion.contains(negFeatureIdx));
                        break;
                    }
                    */



                    sampleMap.put("user", userIdx);
                    sampleMap.put("item", itemIdx);
                    sampleMap.put("feature", featureIdx);
                    sampleSet.add(sampleMap);
                    userSample.put(userIdx, sampleSize);
                    itemSample.put(itemIdx, sampleSize);
                    featureSample.put(featureIdx, sampleSize);

                    itemUserPosFeatureUnion.add(featureIdx);
                    sampleSize++;

                }
                //update hiddenUserFeatureMatrix, explicitUserFeatureMatrix(H1, U1)
                Multimap<Integer, Integer> userSample = sampleIndices.get("user");
                for (Integer userIdx : userSample.keySet()) {
                    Collection<Integer> pairs = userSample.get(userIdx);
                    //H1 init
                    VectorBasedDenseVector itemRatingsVector = new VectorBasedDenseVector(pairs.size());
                    VectorBasedDenseVector itemPredictsVector = new VectorBasedDenseVector(pairs.size());
                    DenseMatrix batchItemHiddenMatrix = new DenseMatrix(pairs.size(), hiddenFeatureNum);

                    //U1 init
                    VectorBasedDenseVector attentionRatingsVector = new VectorBasedDenseVector(pairs.size());
                    VectorBasedDenseVector attentionPredictsVector = new VectorBasedDenseVector(pairs.size());
                    DenseMatrix batchFeatureMatrix = new DenseMatrix(pairs.size(), explicitFeatureNum);
                    DenseMatrix batchItemFeatureMatrix = new DenseMatrix(pairs.size(), explicitFeatureNum);

                    int index = 0;
                    int userFeatureVecIdx = 0;
                    for (Integer sampleId : pairs) {
                        int itemIdx = sampleSet.get(sampleId).get("item");
                        int posFeatureIdx = sampleSet.get(sampleId).get("feature");

                        itemRatingsVector.set(index, trainMatrix.get(userIdx, itemIdx));
                        itemPredictsVector.set(index, predictWithoutBound(userIdx, itemIdx));
                        batchItemHiddenMatrix.set(index, itemHiddenMatrix.row(itemIdx));
                        batchItemFeatureMatrix.set(index, itemFeatureMatrix.row(itemIdx));
                        double lossError = (trainMatrix.get(userIdx, itemIdx) - predictWithoutBound(userIdx, itemIdx));
                        loss += lossError * lossError;
                        index++;


                        if (posFeatureIdx != -1) {
                           attentionRatingsVector.set(userFeatureVecIdx, userFeatureAttention.get(userIdx, posFeatureIdx));
                           attentionPredictsVector.set(userFeatureVecIdx, predUserAttention(userIdx, posFeatureIdx));
                           batchFeatureMatrix.set(userFeatureVecIdx, featureMatrix.row(posFeatureIdx));
                           double attentionLossError = userFeatureAttention.get(userIdx, posFeatureIdx) - predUserAttention(userIdx, posFeatureIdx);
                           loss += attentionLossError * attentionLossError;
                           userFeatureVecIdx++;
                        }
                        else
                            continue;
                    }
                    //calc optimization h1
                    for (int factorIdx = 0; factorIdx < hiddenFeatureNum; factorIdx++) {
                        MatrixBasedDenseVector hiddenItemsVector = (MatrixBasedDenseVector) batchItemHiddenMatrix.column(factorIdx);
                        double realRatingValue = hiddenItemsVector.dot(itemRatingsVector);
                        double estmRatingValue = hiddenItemsVector.dot(itemPredictsVector);
                        double userHiddenValue = userHiddenMatrix.get(userIdx, factorIdx);
                        double error = (realRatingValue - estmRatingValue) - lambdaH * userHiddenValue;

                        userHiddenMatrixLearnRate[userIdx][factorIdx] += error * error;
                        double del = adagrad(userHiddenMatrixLearnRate[userIdx][factorIdx], error, pairs.size());
                        userHiddenMatrix.plus(userIdx, factorIdx, del);
                        if (userHiddenMatrix.get(userIdx, factorIdx) < 0)
                            userHiddenMatrix.set(userIdx, factorIdx, 0.0);
                        loss += lambdaH * userHiddenValue * userHiddenValue;
                    }
                    //calc optimization u1
                    if (userFeatureVecIdx + index == 0)
                        continue;
                    for (int factorIdx = 0; factorIdx < explicitFeatureNum; factorIdx++) {
                        MatrixBasedDenseVector explicitItemsVector = (MatrixBasedDenseVector) batchItemFeatureMatrix.column(factorIdx);
                        MatrixBasedDenseVector featureFactorsVector = (MatrixBasedDenseVector) batchFeatureMatrix.column(factorIdx);

                        double realRatingValue = explicitItemsVector.dot(itemRatingsVector);
                        double estmRatingValue = explicitItemsVector.dot(itemPredictsVector);
                        double realAttentionValue = featureFactorsVector.dot(attentionRatingsVector);
                        double estmAttentionValue = featureFactorsVector.dot(attentionPredictsVector);
                        double userFeatureValue = userFeatureMatrix.get(userIdx, factorIdx);
                        double error = (realRatingValue - estmRatingValue) + lambdaX * (realAttentionValue - estmAttentionValue) - lambdaU * userFeatureValue;
                        userFeatureMatrixLearnRate[userIdx][factorIdx] += error * error;
                        double del = adagrad(userFeatureMatrixLearnRate[userIdx][factorIdx], error, userFeatureVecIdx + index);
                        userFeatureMatrix.plus(userIdx, factorIdx, del);
                        if (userFeatureMatrix.get(userIdx, factorIdx) < 0.0)
                            userFeatureMatrix.set(userIdx, factorIdx, 0.0);
                        loss += lambdaU * userFeatureValue * userFeatureValue;
                    }
                }

                //update hiddenItemFeatureMatrix, explicitItemFeatureMatrix(H2, U2)
                Multimap<Integer, Integer> itemSample = sampleIndices.get("item");
                for (Integer itemIdx : itemSample.keySet()) {
                    Collection<Integer> pairs = itemSample.get(itemIdx);
                    VectorBasedDenseVector userRatingsVector = new VectorBasedDenseVector(pairs.size());
                    VectorBasedDenseVector userPredictsVector = new VectorBasedDenseVector(pairs.size());
                    DenseMatrix batchUserHiddenMatrix = new DenseMatrix(pairs.size(), hiddenFeatureNum);

                    VectorBasedDenseVector itemFeatureRatingsVector = new VectorBasedDenseVector(pairs.size());
                    VectorBasedDenseVector itemFeaturePredictsVector = new VectorBasedDenseVector(pairs.size());
                    DenseMatrix batchUserFeatureMatrix = new DenseMatrix(pairs.size(), explicitFeatureNum);
                    DenseMatrix batchFeatureMatrix = new DenseMatrix(pairs.size(), explicitFeatureNum);

                    int index = 0;
                    int itemFeatureVecIdx = 0;
                    for (Integer sampleId : pairs) {
                        int userIdx = sampleSet.get(sampleId).get("user");
                        int featureIdx = sampleSet.get(sampleId).get("feature");
                        userRatingsVector.set(index, trainMatrix.get(userIdx, itemIdx));
                        userPredictsVector.set(index, predictWithoutBound(userIdx, itemIdx));
                        batchUserHiddenMatrix.set(index, userHiddenMatrix.row(userIdx));
                        batchUserFeatureMatrix.set(index, userFeatureMatrix.row(userIdx));
                        index++;

                        if (featureIdx == -1)
                            continue;
                        itemFeatureRatingsVector.set(itemFeatureVecIdx, itemFeatureQuality.get(itemIdx, featureIdx));
                        itemFeaturePredictsVector.set(itemFeatureVecIdx, predItemQuality(itemIdx, featureIdx));
                        batchFeatureMatrix.set(itemFeatureVecIdx, featureMatrix.row(featureIdx));
                        itemFeatureVecIdx++;
                        double lossError = (itemFeatureQuality.get(itemIdx, featureIdx) - predItemQuality(itemIdx, featureIdx));
                        loss += lossError * lossError;
                    }

                    //calc optimization h2
                    for (int factorIdx = 0; factorIdx < hiddenFeatureNum; factorIdx++) {
                        MatrixBasedDenseVector hiddenUsersVector = (MatrixBasedDenseVector) batchUserHiddenMatrix.column(factorIdx);
                        double realRatingValue = hiddenUsersVector.dot(userRatingsVector);
                        double estmRatingValue = hiddenUsersVector.dot(userPredictsVector);
                        double itemHiddenValue = itemHiddenMatrix.get(itemIdx, factorIdx);
                        double error = (realRatingValue - estmRatingValue) - lambdaH * itemHiddenValue;

                        itemHiddenMatrixLearnRate[itemIdx][factorIdx] += error * error;
                        double del = adagrad(itemHiddenMatrixLearnRate[itemIdx][factorIdx], error, pairs.size());
                        itemHiddenMatrix.plus(itemIdx, factorIdx, del);
                        if (itemHiddenMatrix.get(itemIdx, factorIdx) < 0)
                            itemHiddenMatrix.set(itemIdx, factorIdx, 0.0);
                        loss += lambdaH * itemHiddenValue * itemHiddenValue;
                    }
                    if (index + itemFeatureVecIdx == 0)
                        continue;
                    //calc optimization u2
                    for (int factorIdx = 0; factorIdx < explicitFeatureNum; factorIdx++) {
                        MatrixBasedDenseVector userFeaturesVector = (MatrixBasedDenseVector) batchUserFeatureMatrix.column(factorIdx);
                        MatrixBasedDenseVector featuresVector = (MatrixBasedDenseVector) batchFeatureMatrix.column(factorIdx);
                        double realRatingValue = userFeaturesVector.dot(userRatingsVector);
                        double estmRatingValue = userFeaturesVector.dot(userPredictsVector);
                        double realQualityValue = featuresVector.dot(itemFeatureRatingsVector);
                        double estmQualityValue = featuresVector.dot(itemFeaturePredictsVector);
                        double itemFeatureValue = itemFeatureMatrix.get(itemIdx, factorIdx);

                        double error = (realRatingValue - estmRatingValue) + lambdaY * (realQualityValue - estmQualityValue) - lambdaU * itemFeatureValue;
                        itemFeatureMatrixLearnRate[itemIdx][factorIdx] += error * error;
                        double del = adagrad(itemFeatureMatrixLearnRate[itemIdx][factorIdx], error, index + itemFeatureVecIdx);
                        itemFeatureMatrix.plus(itemIdx, factorIdx, del);
                        if (itemFeatureMatrix.get(itemIdx, factorIdx) < 0)
                            itemFeatureMatrix.set(itemIdx, factorIdx, 0.0);
                        loss += lambdaU * itemFeatureValue * itemFeatureValue;
                    }
                }

                //update posFeatureMatrix
                Multimap<Integer, Integer> featureSample = sampleIndices.get("feature");
                for (Integer posFeatureIdx : featureSample.keySet()) {
                    if (posFeatureIdx == -1)
                        continue;
                    Collection<Integer> pairs = featureSample.get(posFeatureIdx);
                    VectorBasedDenseVector attentionRatingsVector = new VectorBasedDenseVector(pairs.size());
                    VectorBasedDenseVector attentionPredictsVector = new VectorBasedDenseVector(pairs.size());
                    VectorBasedDenseVector qualityRatingsVector = new VectorBasedDenseVector(pairs.size());
                    VectorBasedDenseVector qualityPredictsVector = new VectorBasedDenseVector(pairs.size());
                    DenseMatrix batchUserFeatureMatrix = new DenseMatrix(pairs.size(), explicitFeatureNum);
                    DenseMatrix batchItemFeatureMatrix = new DenseMatrix(pairs.size(), explicitFeatureNum);
                    int userVecIdx = 0;
                    int itemVecIdx = 0;
                    for (Integer sampleId : pairs) {
                        int userIdx = sampleSet.get(sampleId).get("user");
                        int itemIdx = sampleSet.get(sampleId).get("item");

                        double attentionRatingValue = userFeatureAttention.get(userIdx, posFeatureIdx);
                        if (attentionRatingValue != 0.0) {
                            attentionRatingsVector.set(userVecIdx, attentionRatingValue);
                            attentionPredictsVector.set(userVecIdx, predUserAttention(userIdx, posFeatureIdx));
                            batchUserFeatureMatrix.set(userVecIdx, userFeatureMatrix.row(userIdx));
                            userVecIdx++;
                        }


                        double qualityRatingValue = itemFeatureQuality.get(itemIdx, posFeatureIdx);
                        if (qualityRatingValue != 0.0) {
                            qualityRatingsVector.set(itemVecIdx, qualityRatingValue);
                            qualityPredictsVector.set(itemVecIdx, predItemQuality(itemIdx, posFeatureIdx));
                            batchItemFeatureMatrix.set(itemVecIdx, itemFeatureMatrix.row(itemIdx));
                            itemVecIdx++;
                        }
                    }

                    if (userVecIdx + itemVecIdx == 0)
                        continue;
                    for (int factorIdx = 0; factorIdx < explicitFeatureNum; factorIdx++) {
                        MatrixBasedDenseVector featureUsersVector = (MatrixBasedDenseVector) batchUserFeatureMatrix.column(factorIdx);
                        MatrixBasedDenseVector featureItemsVector = (MatrixBasedDenseVector) batchItemFeatureMatrix.column(factorIdx);
                        double posFeatureFactorValue = featureMatrix.get(posFeatureIdx, factorIdx);
                        double realAttentionRatingValue = featureUsersVector.dot(attentionRatingsVector);
                        double estmAttentionRatingValue = featureUsersVector.dot(attentionPredictsVector);
                        double realQualityRatingValue = featureItemsVector.dot(qualityRatingsVector);
                        double estmQualityRatingValue = featureItemsVector.dot(qualityPredictsVector);
                        double error = lambdaX * (realAttentionRatingValue - estmAttentionRatingValue) + lambdaY * (realQualityRatingValue - estmQualityRatingValue) - lambdaV * posFeatureFactorValue;
                        featureMatrixLearnRate[posFeatureIdx][factorIdx] += error * error;
                        double del = adagrad(featureMatrixLearnRate[posFeatureIdx][factorIdx], error, userVecIdx + itemVecIdx);
                        if (Double.isInfinite(del))
                            System.out.println("");
                        featureMatrix.plus(posFeatureIdx, factorIdx, del);
                        if (featureMatrix.get(posFeatureIdx, factorIdx) < 0.0)
                            featureMatrix.set(posFeatureIdx, factorIdx, 0.0);
                        loss += lambdaV * posFeatureFactorValue * posFeatureFactorValue;
                    }
                }
            }
            LOG.info("iter:" + iter + ", loss:" + loss);
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

    protected List<Set<Integer>> getUnion(List<Set<Integer>> leftList, List<Set<Integer>> rightList) {
        List<Set<Integer>> unionList = new ArrayList<>();
        for (int idx = 0; idx < Math.max(leftList.size(), rightList.size()); idx++) {
            Set<Integer> left = leftList.get(idx);
            Set<Integer> union = new HashSet<>(left);
            union.addAll(rightList.get(idx));
            unionList.add(new HashSet<>(union));
        }
        return unionList;
    }

    class Sampling {
        ArrayList<Multimap<Integer, Integer>[]> attribute;
        ArrayList<String> sampleSet;
        int attributeSize;
        HashBiMap<String, Integer> keys;

        Sampling(HashBiMap<String, Integer> keys, ArrayList<Multimap<Integer, Integer>[]> attribute, ArrayList<String> sampleSet) {
            attributeSize = keys.size();
            this.keys = keys;
            this.attribute = attribute;
            this.sampleSet = sampleSet;
        }
    }
}
