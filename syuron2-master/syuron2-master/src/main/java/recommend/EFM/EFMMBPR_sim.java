package recommend.EFM;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.google.common.collect.BiMap;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.HashBiMap;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.Multimap;
import com.google.common.collect.Table;

import net.librec.common.LibrecException;
import net.librec.math.algorithm.Maths;
import net.librec.math.algorithm.Randoms;
import net.librec.math.structure.DataFrame;
import net.librec.math.structure.DenseMatrix;
import net.librec.math.structure.MatrixBasedDenseVector;
import net.librec.math.structure.SequentialAccessSparseMatrix;
import net.librec.math.structure.TensorEntry;
import net.librec.math.structure.VectorBasedDenseVector;

public class EFMMBPR_sim extends EFMBPRecommender{
    //value is similarity
    protected SequentialAccessSparseMatrix userHelpfulFeatureAttention;
    protected SequentialAccessSparseMatrix sideRatingMatrix;
    protected BiMap<Integer, String> sideRatingPairsMappingData;
    protected Table<Integer, Integer, String> userItemFeaturesTable;
    protected Table<Integer, Integer, String> userItemHelpFeaturesTable;

    protected BiMap<Integer, String> userDict;
    protected Map<Integer, String> userHelpedDict;


    @Override
    protected void setup() throws LibrecException {
        super.setup();
        //featureMatrix.init(0.1);
        sideRatingPairsMappingData = DataFrame.getInnerMapping("side").inverse();
        BiMap<String, Integer> userEncodePairsMappingData = DataFrame.getInnerMapping("user");
        Table<Integer, Integer, Double> sideRatingTable = HashBasedTable.create();
        Table<Integer, Integer, String> userHelpfulFeatureAttentionValuesTable = HashBasedTable.create();
        Table<Integer, Integer, Double> userHelpfulFeatureAttentionTable = HashBasedTable.create();
        double similarityThreshold = conf.getDouble("rec.similarity.threshold", 0.4);
        userHelpedDict = new HashMap<>();
        userItemFeaturesTable = HashBasedTable.create();
        //System.out.println("userItemFeaturesTable1");
        //System.out.println(userItemFeaturesTable);
        for (TensorEntry te : trainTensor) {
            int[] entryKeys = te.keys();
            int userIndex = entryKeys[0];
            //System.out.println("entrykeys");
            //System.out.println(entryKeys);
            int itemIndex = entryKeys[1];
            int featureSentimentPairsIndex = entryKeys[2];
            int sideRatingPairsIndex = entryKeys[3];

            String sideRatingPairsString = sideRatingPairsMappingData.get(sideRatingPairsIndex)
                    .replaceAll("</endperson[0-9]+>", "");
            //System.out.println("sideRatingPairsIndex");
            //System.out.println(sideRatingPairsIndex);
            String featureSentimentPairsString = featureSentimemtPairsMappingData.get(featureSentimentPairsIndex)
                    .replaceAll("</endperson[0-9]+>", "");
            //System.out.println("featureSentimentPairsIndex");
            //System.out.println(featureSentimentPairsIndex);
            //System.out.printf(sideRatingPairsString);
            //System.out.printf("abc");
            //System.out.printf(featureSentimentPairsString);
            if (!sideRatingPairsString.isEmpty() && !featureSentimentPairsString.isEmpty()) {
                String[] fSPList = featureSentimentPairsString.split(" ");
                for (String featureAndSentiment : fSPList) {
                    //System.out.println("featureAndSentiment");
                    //System.out.println(featureAndSentiment);
                    String feature = featureAndSentiment.split(":")[0];
                    //System.out.println("feature");
                    //System.out.println(feature);
                    String[] sRPList = sideRatingPairsString.split(" ");
                    //System.out.println("sRPList");
                    //System.out.println(sRPList);

                    for (String srp : sRPList) {
                        String helpedUserIdxString = srp.split(";")[0];
                        int helpedUserEncodeIdx = Integer.valueOf(userEncodePairsMappingData.get(helpedUserIdxString));
                        int featureIdx = featureDict.get(feature);
                        //System.out.println("feature");
                        //System.out.println(feature);
                        //System.out.println("featureIdx");
                        //System.out.println(featureIdx);
                        //double similarity = Double.valueOf(srp.split(";")[3]);
                        String similarityString = srp.split(";")[3];
                        //System.out.println("similarityString");
                        //System.out.println(similarityString);

                        //userHelpfulFeatureAttentionTable.put(helpedUserEncodeIdx, featureIdx, similarity);
                        if (!userHelpfulFeatureAttentionValuesTable.contains(helpedUserEncodeIdx, featureIdx)) {
                            userHelpfulFeatureAttentionValuesTable.put(helpedUserEncodeIdx, featureIdx, similarityString);
                        } else {
                            userHelpfulFeatureAttentionValuesTable.put(helpedUserEncodeIdx, featureIdx,
                                    userHelpfulFeatureAttentionValuesTable.get(helpedUserEncodeIdx, featureIdx) + ":" + similarityString
                                    );
                        }
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
            //System.out.println("sideRatingPairsList");
            //System.out.println(sideRatingPairsList);
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
            //System.out.println("userItemFeaturesTable2");
            //System.out.println(userItemFeaturesTable);
        }

        //delete helpful feature by similarity
        //System.out.println("userHelpfulFeatureAttentionValuesTable");
        //System.out.println(userHelpfulFeatureAttentionValuesTable);
        for (Table.Cell<Integer, Integer, String> cell : userHelpfulFeatureAttentionValuesTable.cellSet()) {
            int userIdx = cell.getRowKey();
            int featureIdx = cell.getColumnKey();
            String[] similarityValues = cell.getValue().split(":");
            double weightSimValue = 0.0;
            int weightSize = 0;
            for (String simValueStr : similarityValues) {
                double simValue = Double.valueOf(simValueStr);
                if (simValue > 0.0) {
                    weightSimValue += simValue;
                    weightSize++;
                }
            }
            if (weightSimValue != 0.0 && weightSize!= 0) {
                double aveSimValue = weightSimValue / (double) weightSize;
                if (aveSimValue > similarityThreshold)
                    userHelpfulFeatureAttentionTable.put(userIdx, featureIdx, aveSimValue);
            }

        }

        userHelpfulFeatureAttention = new SequentialAccessSparseMatrix(numUsers, numberOfFeatures, userHelpfulFeatureAttentionTable);
    }

    protected void featurerank(int userIdx) throws LibrecException{
    	//get feature and values
    	double[] userFeatureValues = featureMatrix.times(userFeatureMatrix.row(userIdx)).getValues();
    	Map<Integer, Double> userFeatureValueMap = new HashMap<>();
    	for (int i = 0; i < numberOfFeatures; i++) {
    		userFeatureValueMap.put(i,userFeatureValues[i]);
    	}


    	//sort features by values
    	userFeatureValueMap = sortByValue(userFeatureValueMap);

    	int numFeatureToExplain = 10;
    	Object[] userTopFeatureIndices = Arrays.copyOfRange(userFeatureValueMap.keySet().toArray(), numberOfFeatures - numFeatureToExplain, numberOfFeatures);
    	String[] userTopFeatureIds = new String[numFeatureToExplain];
    	double[] userTopFeatureValues = new double[numFeatureToExplain];
    	for (int i = 0; i < numFeatureToExplain; i++) {
    		int featureIdx = (int) userTopFeatureIndices[numFeatureToExplain-1-i];
    		userTopFeatureValues[i] = userFeatureValues[featureIdx];
    		userTopFeatureIds[i]=featureDict.inverse().get(featureIdx);
    	}
    	//get features ranking
    	StringBuilder userFeatureSb = new StringBuilder();
    	for (int i = 0; i < numFeatureToExplain; i++) {
    		userFeatureSb.append(userTopFeatureIds[i]).append(":").append(normalize(userTopFeatureValues[i])).append("\n");
    	}
    	LOG.info("user"+ userIdx + "'s most cared features are \n" + userFeatureSb);

    }

    protected void explain(int userIdx) throws LibrecException {
        // get useridx and itemidices
        //int userIdx = userMappingData.get(userId);
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
        LOG.info("user " + userIdx + "'s most cared features are \n" + userFeatureSb);
        LOG.info("item " + recommendedItemId + "'s feature values are\n" + recItemFeatureSb);
        LOG.info("item " + disRecommendedItemId + "'s feature values are\n" + disRecItemFeatureSb);
        LOG.info("So we recommend item " + recommendedItemId + ", disRecommend item " + disRecommendedItemId + " to user " + userIdx);
        LOG.info("___________________________");
    }


    @Override
    protected void trainModel() throws LibrecException {
        //adagrad(sgd)
        epsilon = conf.getDouble("rec.sgd.epsilon", 1e-8);
        eta = conf.getDouble("rec.sgd.eta", 1.0);
        batchSize = conf.getInt("rec.sgd.batchSize", 300);
        double[][] featureMatrixLearnRate = new double[numberOfFeatures][explicitFeatureNum];
        //System.out.println("featureMatrixLearnRate");
        //System.out.println(featureMatrixLearnRate);
        double[][] userFeatureMatrixLearnRate = new double[numUsers][explicitFeatureNum];
        double[][] userHiddenMatrixLearnRate = new double[numUsers][numFactors - explicitFeatureNum];
        double[][] itemFeatureMatrixLearnRate = new double[numItems][explicitFeatureNum];
        double[][] itemHiddenMatrixLearnRate = new double[numItems][numFactors - explicitFeatureNum];
        List<Set<Integer>> userItemsSet = getRowColumnsSet(trainMatrix, numUsers);
        System.out.println("userItemsSet");
        System.out.println(userItemsSet);
        List<Set<Integer>> userFeaturesSet = getRowColumnsSet(userFeatureAttention, numUsers);
        System.out.println("userFeaturesSet");
        System.out.println(userFeaturesSet);
        List<Set<Integer>> itemFeaturesSet = getRowColumnsSet(itemFeatureQuality, numItems);
        System.out.println("itemFeaturesSet");
        System.out.println(itemFeaturesSet);
        List<Set<Integer>> userHelpFeaturesSet = getRowColumnsSet(userHelpfulFeatureAttention, numUsers);
        System.out.println("userHelpFeaturesSet");
        System.out.println(userHelpFeaturesSet);
        List<Set<Integer>> userFeaturesUnion = getUnion(userFeaturesSet, userHelpFeaturesSet);
        System.out.println("userFeaturesUnioin");
        System.out.println(userFeaturesUnion);
        List<Set<Integer>> subUserHelpFeaturesSet = getAnd(userHelpFeaturesSet, userFeaturesSet);
        System.out.println("subUserHelpFeaturesSet");
        System.out.println(subUserHelpFeaturesSet);

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
                sampleIndices.put("negFeature", HashMultimap.create());
                sampleIndices.put("userHelp", HashMultimap.create());
                sampleIndices.put("itemHelp", HashMultimap.create());
                sampleIndices.put("featureHelp", HashMultimap.create());
                sampleIndices.put("negFeatureHelp", HashMultimap.create());
                sampleIndices.put("itemFeature", HashMultimap.create());

                Set<Integer> itemUserPosFeatureUnion = new HashSet<>();
                Set<Integer> itemUserNegFeatureUnion = new HashSet<>();
                Set<Integer> itemUserPosFeatureHelpUnion = new HashSet<>();
                Set<Integer> itemUserNegFeatureHelpUnion = new HashSet<>();
                int sampleSize = 0;
                int qualityToHelpSize = 0;
                while (sampleSize < batchSize) {
                    Map<String, Integer> sampleMap = new HashMap<>();
                    Multimap<Integer, Integer> userSample = sampleIndices.get("user");
                    Multimap<Integer, Integer> itemSample = sampleIndices.get("item");
                    Multimap<Integer, Integer> featureSample = sampleIndices.get("feature");
                    Multimap<Integer, Integer> negFeatureSample = sampleIndices.get("negFeature");
                    Multimap<Integer, Integer> featureHelpSample = sampleIndices.get("featureHelp");
                    Multimap<Integer, Integer> negFeatureHelpSample = sampleIndices.get("negFeatureHelp");
                    int userIdx = Randoms.uniform(numUsers);
                    Set<Integer> itemsSet = userItemsSet.get(userIdx);
                    if (itemsSet.size() == 0 || itemsSet.size() == numItems)
                        continue;
                    int[] itemIndices = trainMatrix.row(userIdx).getIndices();
                    int itemIdx = itemIndices[Randoms.uniform(itemIndices.length)];

                    //user feature, negative feature sampling
                    int featureIdx = -1;
                    int negFeatureIdx = -1;
                    if (userItemFeaturesTable.contains(userIdx, itemIdx)) {
                        Set<Integer> uFUnion = userFeaturesUnion.get(userIdx);
                        String[] featuresList = userItemFeaturesTable.get(userIdx, itemIdx).split(" ");
                        String featureString = featuresList[Randoms.uniform(featuresList.length)].split(":")[0];
                        featureIdx = featureDict.get(featureString);
                        do {
                            negFeatureIdx = Randoms.uniform(numberOfFeatures);
                        } while (uFUnion.contains(negFeatureIdx));
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


                    //helpful sampling
                    int userHelpIdx, itemHelpIdx, featureHelpIdx, negFeatureHelpIdx;
                    userHelpIdx = itemHelpIdx = featureHelpIdx = negFeatureHelpIdx = -1;

                    //featureHelp negFeatureHelp sampling from sampling helpful review
                    while (true) {
                        Set<Integer> subUFHSet = subUserHelpFeaturesSet.get(userIdx);
                        Set<Integer> uFUnion = userFeaturesUnion.get(userIdx);

                        if (subUFHSet.size() == 0 || subUFHSet.size() == numberOfFeatures)
                            break;
                        int[] helpFeatureIndices = userHelpfulFeatureAttention.row(userIdx).getIndices();
                        featureHelpIdx = helpFeatureIndices[Randoms.uniform(helpFeatureIndices.length)];
                        if (itemFeatureQuality.get(itemIdx, featureHelpIdx) != 0.0)
                            qualityToHelpSize++;
                        do {
                            negFeatureHelpIdx = Randoms.uniform(numberOfFeatures);
                        } while (uFUnion.contains(negFeatureHelpIdx));
                        break;
                    }

                    sampleMap.put("user", userIdx);
                    sampleMap.put("item", itemIdx);
                    sampleMap.put("feature", featureIdx);
                    sampleMap.put("negFeature", negFeatureIdx);
                    sampleMap.put("featureHelp", featureHelpIdx);
                    sampleMap.put("negFeatureHelp", negFeatureHelpIdx);
                    sampleSet.add(sampleMap);
                    userSample.put(userIdx, sampleSize);
                    itemSample.put(itemIdx, sampleSize);
                    featureSample.put(featureIdx, sampleSize);
                    negFeatureSample.put(negFeatureIdx, sampleSize);
                    featureHelpSample.put(featureHelpIdx, sampleSize);
                    negFeatureHelpSample.put(negFeatureHelpIdx, sampleSize);

                    itemUserPosFeatureUnion.add(featureIdx);
                    itemUserNegFeatureUnion.add(negFeatureIdx);
                    itemUserPosFeatureHelpUnion.add(featureHelpIdx);
                    itemUserNegFeatureHelpUnion.add(negFeatureHelpIdx);
                    sampleSize++;

                }
                //LOG.info("feature help samples for quality is " + qualityToHelpSize);
                //update hiddenUserFeatureMatrix, explicitUserFeatureMatrix(H1, U1)
                Multimap<Integer, Integer> userSample = sampleIndices.get("user");
                for (Integer userIdx : userSample.keySet()) {
                    Collection<Integer> pairs = userSample.get(userIdx);
                    //H1 init
                    VectorBasedDenseVector itemRatingsVector = new VectorBasedDenseVector(pairs.size());
                    VectorBasedDenseVector itemPredictsVector = new VectorBasedDenseVector(pairs.size());
                    DenseMatrix batchItemHiddenMatrix = new DenseMatrix(pairs.size(), hiddenFeatureNum);

                    //U1 init
                    VectorBasedDenseVector deriValuesVector = new VectorBasedDenseVector(pairs.size());
                    DenseMatrix batchFeatureMatrix = new DenseMatrix(pairs.size(), explicitFeatureNum);
                    DenseMatrix batchItemFeatureMatrix = new DenseMatrix(pairs.size(), explicitFeatureNum);

                    int index = 0;
                    int userFeatureVecIdx = 0;
                    for (Integer sampleId : pairs) {
                        int itemIdx = sampleSet.get(sampleId).get("item");
                        int posFeatureIdx = sampleSet.get(sampleId).get("feature");
                        int negFeatureIdx = sampleSet.get(sampleId).get("negFeature");
                        int posFeatureHelpIdx = sampleSet.get(sampleId).get("featureHelp");
                        int negFeatureHelpIdx = sampleSet.get(sampleId).get("negFeatureHelp");

                        itemRatingsVector.set(index, trainMatrix.get(userIdx, itemIdx));
                        itemPredictsVector.set(index, predictWithoutBound(userIdx, itemIdx));
                        batchItemHiddenMatrix.set(index, itemHiddenMatrix.row(itemIdx));
                        batchItemFeatureMatrix.set(index, itemFeatureMatrix.row(itemIdx));
                        double lossError = (trainMatrix.get(userIdx, itemIdx) - predictWithoutBound(userIdx, itemIdx));
                        loss += lossError * lossError;
                        index++;


                        double diffValue;
                        if (posFeatureIdx != -1 && posFeatureHelpIdx != -1) {
                            double posPredictRating = predUserAttention(userIdx, posFeatureIdx);
                            double negPredictRating = predUserAttention(userIdx, negFeatureIdx);
                            double posHelpPredictRating = predUserAttention(userIdx, posFeatureHelpIdx);
                            double negHelpPredictRating = predUserAttention(userIdx, negFeatureHelpIdx);
                            diffValue = (posPredictRating - negPredictRating) - (posHelpPredictRating - negHelpPredictRating);
                            for (int factorIdx = 0; factorIdx < explicitFeatureNum; factorIdx++) {
                                batchFeatureMatrix.set(userFeatureVecIdx, factorIdx, featureMatrix.get(posFeatureIdx, factorIdx) - featureMatrix.get(negFeatureIdx, factorIdx)
                                        - (featureMatrix.get(posFeatureHelpIdx, factorIdx) - featureMatrix.get(negFeatureHelpIdx, factorIdx)));
                            }
                        }
                        else if (posFeatureIdx == -1 && posFeatureHelpIdx != -1) {
                            double posHelpPredictRating = predUserAttention(userIdx, posFeatureHelpIdx);
                            double negHelpPredictRating = predUserAttention(userIdx, negFeatureHelpIdx);
                            diffValue = posHelpPredictRating - negHelpPredictRating;
                            for (int factorIdx = 0; factorIdx < explicitFeatureNum; factorIdx++) {
                                batchFeatureMatrix.set(userFeatureVecIdx, factorIdx,  featureMatrix.get(posFeatureHelpIdx, factorIdx) - featureMatrix.get(negFeatureHelpIdx, factorIdx));
                            }
                        }
                        else if (posFeatureIdx != - 1 && posFeatureHelpIdx == -1) {
                            double posPredictRating = predUserAttention(userIdx, posFeatureIdx);
                            double negPredictRating = predUserAttention(userIdx, negFeatureIdx);
                            diffValue = posPredictRating - negPredictRating;
                            for (int factorIdx = 0; factorIdx < explicitFeatureNum; factorIdx++) {
                                batchFeatureMatrix.set(userFeatureVecIdx, factorIdx, featureMatrix.get(posFeatureIdx, factorIdx) - featureMatrix.get(negFeatureIdx, factorIdx));
                            }
                        }
                        else
                            continue;

                        double lossValue = -Math.log(Maths.logistic(diffValue));
                        loss+= lossValue;
                        double deriValue = Maths.logistic(-diffValue);
                        deriValuesVector.set(userFeatureVecIdx, deriValue);
                        userFeatureVecIdx++;
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
                        double userFeatureValue = userFeatureMatrix.get(userIdx, factorIdx);
                        double estmDeriValue = featureFactorsVector.dot(deriValuesVector);
                        double error = (realRatingValue - estmRatingValue) + lambdaX * estmDeriValue   - lambdaU * userFeatureValue;
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
                    VectorBasedDenseVector deriValuesVector = new VectorBasedDenseVector(pairs.size());
                    VectorBasedDenseVector itemRatingsVector = new VectorBasedDenseVector(pairs.size());
                    VectorBasedDenseVector itemPredictsVector = new VectorBasedDenseVector(pairs.size());
                    DenseMatrix batchUserFeatureMatrix = new DenseMatrix(pairs.size(), explicitFeatureNum);
                    DenseMatrix batchItemFeatureMatrix = new DenseMatrix(pairs.size(), explicitFeatureNum);
                    int userVecIdx = 0;
                    int itemVecIdx = 0;
                    for (Integer sampleId : pairs) {
                        int userIdx = sampleSet.get(sampleId).get("user");
                        int itemIdx = sampleSet.get(sampleId).get("item");
                        int negFeatureIdx = sampleSet.get(sampleId).get("negFeature");
                        int posFeatureHelpIdx = sampleSet.get(sampleId).get("featureHelp");
                        int negFeatureHelpIdx = sampleSet.get(sampleId).get("negFeatureHelp");

                        double diffValue;
                        if (posFeatureHelpIdx != -1) {
                            double posPredictRating = predUserAttention(userIdx, posFeatureIdx);
                            double negPredictRating = predUserAttention(userIdx, negFeatureIdx);
                            double posHelpPredictRating = predUserAttention(userIdx, posFeatureHelpIdx);
                            double negHelpPredictRating = predUserAttention(userIdx, negFeatureHelpIdx);
                            diffValue = (posPredictRating - negPredictRating) - (posHelpPredictRating - negHelpPredictRating);
                        } else {
                            double posPredictRating = predUserAttention(userIdx, posFeatureIdx);
                            double negPredictRating = predUserAttention(userIdx, negFeatureIdx);
                            diffValue = (posPredictRating - negPredictRating);
                        }
                        //double lossValue = -Math.log(Maths.logistic(diffValue));
                        //loss += lossValue;

                        double deriValue = Maths.logistic(-diffValue);
                        deriValuesVector.set(userVecIdx, deriValue);
                        batchUserFeatureMatrix.set(userVecIdx, userFeatureMatrix.row(userIdx));

                        double itemRatingValue = itemFeatureQuality.get(itemIdx, posFeatureIdx);
                        if (itemRatingValue != 0.0) {
                            itemRatingsVector.set(itemVecIdx, itemRatingValue);
                            itemPredictsVector.set(itemVecIdx, predItemQuality(itemIdx, posFeatureIdx));
                            batchItemFeatureMatrix.set(itemVecIdx, itemFeatureMatrix.row(itemIdx));
                            itemVecIdx++;
                        }
                        userVecIdx++;
                    }

                    if (userVecIdx + itemVecIdx == 0)
                        continue;
                    for (int factorIdx = 0; factorIdx < explicitFeatureNum; factorIdx++) {
                        MatrixBasedDenseVector featureUsersVector = (MatrixBasedDenseVector) batchUserFeatureMatrix.column(factorIdx);
                        MatrixBasedDenseVector featureItemsVector = (MatrixBasedDenseVector) batchItemFeatureMatrix.column(factorIdx);
                        double posFeatureFactorValue = featureMatrix.get(posFeatureIdx, factorIdx);
                        double estmDiffValue = deriValuesVector.dot(featureUsersVector);
                        double realItemRatingValue = featureItemsVector.dot(itemRatingsVector);
                        double estmItemRatingValue = featureItemsVector.dot(itemPredictsVector);
                        double error = lambdaX * estmDiffValue + lambdaY * (realItemRatingValue - estmItemRatingValue) - lambdaV * posFeatureFactorValue;
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

                //update featureMatrix(neg)
                Multimap<Integer, Integer> negFeatureSample = sampleIndices.get("negFeature");
                for (Integer negFeatureIdx : negFeatureSample.keySet()) {
                    if (negFeatureIdx == -1)
                        continue;
                    Collection<Integer> pairs = negFeatureSample.get(negFeatureIdx);
                    VectorBasedDenseVector deriValuesVector = new VectorBasedDenseVector(pairs.size());
                    VectorBasedDenseVector itemRatingsVector = new VectorBasedDenseVector(pairs.size());
                    VectorBasedDenseVector itemPredictsVector = new VectorBasedDenseVector(pairs.size());
                    DenseMatrix batchUserFeatureMatrix = new DenseMatrix(pairs.size(), explicitFeatureNum);
                    DenseMatrix batchItemFeatureMatrix = new DenseMatrix(pairs.size(), explicitFeatureNum);
                    int userVecIdx = 0;
                    int itemVecIdx = 0;
                    for (Integer sampleId : pairs) {
                        int userIdx = sampleSet.get(sampleId).get("user");
                        int itemIdx = sampleSet.get(sampleId).get("item");
                        int posFeatureIdx = sampleSet.get(sampleId).get("feature");
                        if (posFeatureIdx == -1)
                            continue;
                        int posFeatureHelpIdx = sampleSet.get(sampleId).get("featureHelp");
                        int negFeatureHelpIdx = sampleSet.get(sampleId).get("negFeatureHelp");
                        double diffValue;
                        if (posFeatureHelpIdx != -1) {
                            double posPredictRating = predUserAttention(userIdx, posFeatureIdx);
                            //System.out.println();
                            //System.out.println()
;                            double negPredictRating = predUserAttention(userIdx, negFeatureIdx);
                            double posHelpPredictRating = predUserAttention(userIdx, posFeatureHelpIdx);
                            double negHelpPredictRating = predUserAttention(userIdx, negFeatureHelpIdx);

                            diffValue = (posPredictRating - negPredictRating) - (posHelpPredictRating - negHelpPredictRating);
                        } else {
                            double posPredictRating = predUserAttention(userIdx, posFeatureIdx);
                            double negPredictRating = predUserAttention(userIdx, negFeatureIdx);
                            diffValue = (posPredictRating - negPredictRating);
                        }
                        double lossValue = -Math.log(Maths.logistic(diffValue));
                        loss += lossValue;

                        double deriValue = Maths.logistic(-diffValue);
                        deriValuesVector.set(userVecIdx, deriValue);
                        batchUserFeatureMatrix.set(userVecIdx, userFeatureMatrix.row(userIdx));

                        double itemRatingValue = itemFeatureQuality.get(itemIdx, negFeatureIdx);
                        if (itemRatingValue != 0.0) {
                            itemRatingsVector.set(itemVecIdx, itemRatingValue);
                            itemPredictsVector.set(itemVecIdx, predItemQuality(itemIdx, negFeatureIdx));
                            batchItemFeatureMatrix.set(itemVecIdx, itemFeatureMatrix.row(itemIdx));
                            itemVecIdx++;
                        }
                        userVecIdx++;
                    }
                    if (userVecIdx + itemVecIdx == 0)
                        continue;
                    for (int factorIdx = 0; factorIdx < explicitFeatureNum; factorIdx++) {
                        MatrixBasedDenseVector featureUsersVector = (MatrixBasedDenseVector) batchUserFeatureMatrix.column(factorIdx);
                        MatrixBasedDenseVector featureItemsVector = (MatrixBasedDenseVector) batchItemFeatureMatrix.column(factorIdx);
                        double negFeatureFactorValue = featureMatrix.get(negFeatureIdx, factorIdx);
                        double estmDiffValue = deriValuesVector.dot(featureUsersVector);
                        double realItemRatingValue = featureItemsVector.dot(itemRatingsVector);
                        double estmItemRatingValue = featureItemsVector.dot(itemPredictsVector);
                        double error = lambdaX * ( - estmDiffValue) + lambdaY * (realItemRatingValue - estmItemRatingValue) - lambdaV * negFeatureFactorValue;
                        featureMatrixLearnRate[negFeatureIdx][factorIdx] += error * error;
                        double del = adagrad(featureMatrixLearnRate[negFeatureIdx][factorIdx], error, userVecIdx + itemVecIdx);
                        if (Double.isInfinite(del))
                            System.out.println("");
                        featureMatrix.plus(negFeatureIdx, factorIdx, del);
                        if (featureMatrix.get(negFeatureIdx, factorIdx) < 0.0)
                            featureMatrix.set(negFeatureIdx, factorIdx, 0.0);
                        loss += lambdaV * negFeatureFactorValue * negFeatureFactorValue;
                    }
                }
                //update featureMatrix(posHelp)
                Multimap<Integer, Integer> posFeatureHelpSample = sampleIndices.get("featureHelp");
                for (Integer posFeatureHelpIdx : posFeatureHelpSample.keySet()) {
                    if (posFeatureHelpIdx == -1)
                        continue;
                    Collection<Integer> pairs = posFeatureHelpSample.get(posFeatureHelpIdx);
                    VectorBasedDenseVector deriValuesVector = new VectorBasedDenseVector(pairs.size());
                    VectorBasedDenseVector itemRatingsVector = new VectorBasedDenseVector(pairs.size());
                    VectorBasedDenseVector itemPredictsVector = new VectorBasedDenseVector(pairs.size());
                    DenseMatrix batchUserFeatureMatrix = new DenseMatrix(pairs.size(), explicitFeatureNum);
                    DenseMatrix batchItemFeatureMatrix = new DenseMatrix(pairs.size(), explicitFeatureNum);
                    int userVecIdx = 0;
                    int itemVecIdx = 0;
                    for (Integer sampleId : pairs) {
                        int userIdx = sampleSet.get(sampleId).get("user");
                        int itemIdx = sampleSet.get(sampleId).get("item");
                        int posFeatureIdx = sampleSet.get(sampleId).get("feature");
                        int negFeatureIdx = sampleSet.get(sampleId).get("negFeature");
                        int negFeatureHelpIdx = sampleSet.get(sampleId).get("negFeatureHelp");

                        double diffValue;
                        if (posFeatureIdx == -1) {
                            double posHelpPredictRating = predUserAttention(userIdx, posFeatureHelpIdx);
                            double negHelpPredictRating = predUserAttention(userIdx, negFeatureHelpIdx);
                            diffValue = (posHelpPredictRating - negHelpPredictRating);
                            for (int factorIdx = 0; factorIdx < explicitFeatureNum; factorIdx++) {
                                batchUserFeatureMatrix.set(userVecIdx, factorIdx, userFeatureMatrix.get(userIdx, factorIdx));
                            }
                        }
                        else {
                            double posPredictRating = predUserAttention(userIdx, posFeatureIdx);
                            double negPredictRating = predUserAttention(userIdx, negFeatureIdx);
                            double posHelpPredictRating = predUserAttention(userIdx, posFeatureHelpIdx);
                            double negHelpPredictRating = predUserAttention(userIdx, negFeatureHelpIdx);
                            diffValue = (posPredictRating - negPredictRating) - (posHelpPredictRating - negHelpPredictRating);
                            for (int factorIdx = 0; factorIdx < explicitFeatureNum; factorIdx++) {
                                batchUserFeatureMatrix.set(userVecIdx, factorIdx, -userFeatureMatrix.get(userIdx, factorIdx));
                            }
                        }


                        double lossValue = -Math.log(Maths.logistic(diffValue));
                        loss += lossValue;

                        double deriValue = Maths.logistic(-diffValue);
                        deriValuesVector.set(userVecIdx, deriValue);
                        //batchUserFeatureMatrix.set(userVecIdx, userFeatureMatrix.row(userIdx));

                        double itemRatingValue = itemFeatureQuality.get(itemIdx, posFeatureHelpIdx);
                        if (itemRatingValue != 0.0) {
                            itemRatingsVector.set(itemVecIdx, itemRatingValue);
                            itemPredictsVector.set(itemVecIdx, predItemQuality(itemIdx, posFeatureHelpIdx));
                            batchItemFeatureMatrix.set(itemVecIdx, itemFeatureMatrix.row(itemIdx));
                            itemVecIdx++;
                        }
                        userVecIdx++;
                    }
                    if (userVecIdx + itemVecIdx == 0)
                        continue;
                    for (int factorIdx = 0; factorIdx < explicitFeatureNum; factorIdx++) {
                        MatrixBasedDenseVector featureUsersVector = (MatrixBasedDenseVector) batchUserFeatureMatrix.column(factorIdx);
                        MatrixBasedDenseVector featureItemsVector = (MatrixBasedDenseVector) batchItemFeatureMatrix.column(factorIdx);
                        double posFeatureHelpFactorValue = featureMatrix.get(posFeatureHelpIdx, factorIdx);
                        double estmDiffValue = deriValuesVector.dot(featureUsersVector);
                        double realItemRatingValue = featureItemsVector.dot(itemRatingsVector);
                        double estmItemRatingValue = featureItemsVector.dot(itemPredictsVector);
                        double error;
                        error = lambdaX * estmDiffValue + lambdaY * (realItemRatingValue - estmItemRatingValue) - lambdaV * posFeatureHelpFactorValue;
                        featureMatrixLearnRate[posFeatureHelpIdx][factorIdx] += error * error;
                        double del = adagrad(featureMatrixLearnRate[posFeatureHelpIdx][factorIdx], error, userVecIdx + itemVecIdx);
                        if (Double.isInfinite(del))
                            System.out.println("");
                        featureMatrix.plus(posFeatureHelpIdx, factorIdx, del);
                        if (featureMatrix.get(posFeatureHelpIdx, factorIdx) < 0.0)
                            featureMatrix.set(posFeatureHelpIdx, factorIdx, 0.0);
                        loss += lambdaV * posFeatureHelpFactorValue * posFeatureHelpFactorValue;
                    }
                }
                //update featureMatrix(negHelp)
                Multimap<Integer, Integer> negFeatureHelpSample = sampleIndices.get("negFeatureHelp");
                for (Integer negFeatureHelpIdx : negFeatureHelpSample.keySet()) {
                    if (negFeatureHelpIdx == -1)
                        continue;
                    Collection<Integer> pairs = negFeatureHelpSample.get(negFeatureHelpIdx);
                    VectorBasedDenseVector deriValuesVector = new VectorBasedDenseVector(pairs.size());
                    VectorBasedDenseVector itemRatingsVector = new VectorBasedDenseVector(pairs.size());
                    VectorBasedDenseVector itemPredictsVector = new VectorBasedDenseVector(pairs.size());
                    DenseMatrix batchUserFeatureMatrix = new DenseMatrix(pairs.size(), explicitFeatureNum);
                    DenseMatrix batchItemFeatureMatrix = new DenseMatrix(pairs.size(), explicitFeatureNum);
                    int userVecIdx = 0;
                    int itemVecIdx = 0;
                    for (Integer sampleId : pairs) {
                        int userIdx = sampleSet.get(sampleId).get("user");
                        int itemIdx = sampleSet.get(sampleId).get("item");
                        int posFeatureIdx = sampleSet.get(sampleId).get("feature");
                        int negFeatureIdx = sampleSet.get(sampleId).get("negFeature");
                        int posFeatureHelpIdx = sampleSet.get(sampleId).get("featureHelp");
                        if (posFeatureHelpIdx == -1)
                            continue;
                        double diffValue;
                        if (posFeatureIdx == -1) {
                            double posHelpPredictRating = predUserAttention(userIdx, posFeatureHelpIdx);
                            double negHelpPredictRating = predUserAttention(userIdx, negFeatureHelpIdx);
                            diffValue = (posHelpPredictRating - negHelpPredictRating);
                            for (int factorIdx = 0; factorIdx < explicitFeatureNum; factorIdx++) {
                                batchUserFeatureMatrix.set(userVecIdx, factorIdx, -userFeatureMatrix.get(userIdx, factorIdx));
                            }
                        }
                        else {
                            double posPredictRating = predUserAttention(userIdx, posFeatureIdx);
                            double negPredictRating = predUserAttention(userIdx, negFeatureIdx);
                            double posHelpPredictRating = predUserAttention(userIdx, posFeatureHelpIdx);
                            double negHelpPredictRating = predUserAttention(userIdx, negFeatureHelpIdx);
                            diffValue = (posPredictRating - negPredictRating) - (posHelpPredictRating - negHelpPredictRating);
                            for (int factorIdx = 0; factorIdx < explicitFeatureNum; factorIdx++) {
                                batchUserFeatureMatrix.set(userVecIdx, factorIdx, userFeatureMatrix.get(userIdx, factorIdx));
                            }
                        }


                        double lossValue = -Math.log(Maths.logistic(diffValue));
                        loss += lossValue;

                        double deriValue = Maths.logistic(-diffValue);
                        deriValuesVector.set(userVecIdx, deriValue);
                        //batchUserFeatureMatrix.set(userVecIdx, userFeatureMatrix.row(userIdx));

                        double itemRatingValue = itemFeatureQuality.get(itemIdx, negFeatureHelpIdx);
                        if (itemRatingValue != 0.0) {
                            itemRatingsVector.set(itemVecIdx, itemRatingValue);
                            itemPredictsVector.set(itemVecIdx, predItemQuality(itemIdx, negFeatureHelpIdx));
                            batchItemFeatureMatrix.set(itemVecIdx, itemFeatureMatrix.row(itemIdx));
                            itemVecIdx++;
                        }
                        userVecIdx++;
                    }
                    if (userVecIdx + itemVecIdx == 0)
                        continue;
                    for (int factorIdx = 0; factorIdx < explicitFeatureNum; factorIdx++) {
                        MatrixBasedDenseVector featureUsersVector = (MatrixBasedDenseVector) batchUserFeatureMatrix.column(factorIdx);
                        MatrixBasedDenseVector featureItemsVector = (MatrixBasedDenseVector) batchItemFeatureMatrix.column(factorIdx);
                        double negFeatureHelpFactorValue = featureMatrix.get(negFeatureHelpIdx, factorIdx);
                        double estmDiffValue = deriValuesVector.dot(featureUsersVector);
                        double realItemRatingValue = featureItemsVector.dot(itemRatingsVector);
                        double estmItemRatingValue = featureItemsVector.dot(itemPredictsVector);
                        double error;
                        error = lambdaX * ( estmDiffValue) + lambdaY * (realItemRatingValue - estmItemRatingValue) - lambdaV * negFeatureHelpFactorValue;
                        featureMatrixLearnRate[negFeatureHelpIdx][factorIdx] += error * error;
                        double del = adagrad(featureMatrixLearnRate[negFeatureHelpIdx][factorIdx], error, userVecIdx + itemVecIdx);
                        if (Double.isInfinite(del))
                            System.out.println("");
                        featureMatrix.plus(negFeatureHelpIdx, factorIdx, del);
                        if (featureMatrix.get(negFeatureHelpIdx, factorIdx) < 0.0)
                            featureMatrix.set(negFeatureHelpIdx, factorIdx, 0.0);
                        loss += lambdaV * negFeatureHelpFactorValue * negFeatureHelpFactorValue;
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
                //System.out.println("userIdx");
                //System.out.println(userIdx);
                //System.out.println("featureIdx");
                //System.out.println(featureIdx);
                //System.out.println("predValue");
                //System.out.println(predValue);
                maxUserFeature = Math.max(predValue, maxUserFeature);
                //System.out.println("maxUserFeature");
                //System.out.println(maxUserFeature);
                minUserFeature = Math.min(predValue, minUserFeature);
                //System.out.println("minUserFeature");
                //System.out.println(minUserFeature);
            }
        }
        //System.out.println("maxUserFeature");
        //System.out.println(maxUserFeature);
        //explain(1);
        int i=0;
        for (i=0;i<50;i++) {
        	featurerank(i);
        }
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

    protected List<Set<Integer>> getAnd(List<Set<Integer>> leftList, List<Set<Integer>> rightList) {
        List<Set<Integer>> subList = new ArrayList<>();
        for (int idx = 0; idx < Math.max(leftList.size(), rightList.size()); idx++) {
            Set<Integer> left = leftList.get(idx);
            Set<Integer> sub = new HashSet<>(left);
            sub.removeAll(rightList.get(idx));
            subList.add(new HashSet<>(sub));
        }
        return subList;
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
