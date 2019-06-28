package recommend.EFM;

import com.google.common.collect.*;
import net.librec.common.LibrecException;
import net.librec.math.algorithm.Randoms;
import net.librec.math.structure.DataFrame;
import net.librec.math.structure.SequentialAccessSparseMatrix;
import net.librec.math.structure.TensorEntry;
import recommend.EfmRecommender;

import java.util.*;

public class EFM extends EfmRecommender {
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




    @Override
    protected void trainModel() throws LibrecException {
        //adagrad(sgd)
        double epsilon = conf.getDouble("rec.sgd.epsilon", 1e-8);
        double eta = conf.getDouble("rec.sgd.eta", 1.0);
        int batchSize = conf.getInt("rec.sgd.batchSize", 300);
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

        for (int iter = 1; iter <= numIterations; iter++) {
            int maxSample = (trainMatrix.size() + userFeatureAttention.size() + itemFeatureQuality.size() + userHelpfulFeatureAttention.size()) / (4 * batchSize);

            loss = 0.0d;
            for (int sampleCount = 0; sampleCount < maxSample; sampleCount++) {
                ArrayList<Map<String, Integer>> sampleSet = new ArrayList<>();
                Map<String, Multimap<Integer, Integer>> sampleIndices = new HashMap<>();
                sampleIndices.put("user", HashMultimap.create());
                sampleIndices.put("item", HashMultimap.create());
                sampleIndices.put("feature", HashMultimap.create());
                sampleIndices.put("featureHelp", HashMultimap.create());

                int sampleSize = 0;
                while (sampleSize < batchSize) {
                    Map<String, Integer> sampleMap = new HashMap<>();
                    Multimap<Integer, Integer> userSample = sampleIndices.get("user");
                    Multimap<Integer, Integer> itemSample = sampleIndices.get("item");
                    Multimap<Integer, Integer> featureSample = sampleIndices.get("feature");
                    Multimap<Integer, Integer> userHelpSample = sampleIndices.get("userHelp");
                    Multimap<Integer, Integer> itemHelpSample = sampleIndices.get("itemHelp");
                    Multimap<Integer, Integer> featureHelpSample = sampleIndices.get("featureHelp");
                    int userIdx = Randoms.uniform(numUsers);
                    Set<Integer> itemsSet = userItemsSet.get(userIdx);
                    if (itemsSet.size() == 0 || itemsSet.size() == numItems)
                        continue;
                    int[] itemIndices = trainMatrix.row(userIdx).getIndices();
                    int itemIdx = itemIndices[Randoms.uniform(itemIndices.length)];

                    //feature sampling
                    int featureIdx = -1;
                    if (userItemFeaturesTable.contains(userIdx, itemIdx)) {
                        String featuresString = userItemFeaturesTable.get(userIdx, itemIdx);
                        String[] featuresList = featuresString.split(" ");
                        featureIdx = featureDict.get(featuresList[Randoms.uniform(featuresList.length)].split(":")[0]);
                    }
                    //helpful sampling
                    int userHelpIdx, itemHelpIdx, featureHelpIdx;
                    userHelpIdx = itemHelpIdx = featureHelpIdx = -1;
                    if (userHelpedDict.containsKey(userIdx)) {
                        String[] helpsString = userHelpedDict.get(userIdx).split(" ");
                        String help = helpsString[Randoms.uniform(helpsString.length)];
                        userHelpIdx = Integer.valueOf(help.split(":")[0]);
                        itemHelpIdx = Integer.valueOf(help.split(":")[1]);

                        //featureHelp sampling
                        if (userItemFeaturesTable.contains(userHelpIdx, itemHelpIdx)) {
                            String featuresString = userItemFeaturesTable.get(userIdx, itemIdx);
                            String[] featuresList = featuresString.split(" ");
                            featureHelpIdx = featureDict.get(featuresList[Randoms.uniform(featuresList.length)].split(":")[0]);
                        }
                    }
                    sampleMap.put("user", userIdx);
                    sampleMap.put("item", itemIdx);
                    sampleMap.put("feature", featureIdx);
                    sampleMap.put("userHelp", userHelpIdx);
                    sampleMap.put("itemHelp", itemHelpIdx);
                    sampleMap.put("featureHelp", featureHelpIdx);
                    sampleSet.add(sampleMap);
                    userSample.put(userIdx, sampleSize);
                    itemSample.put(itemIdx, sampleSize);
                    featureSample.put(featureIdx, sampleSize);
                    userHelpSample.put(userHelpIdx, sampleSize);
                    itemHelpSample.put(itemHelpIdx, sampleSize);
                    featureHelpSample.put(featureHelpIdx, sampleSize);
                    sampleSize++;
                }

            }
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

}
