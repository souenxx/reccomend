package recommend;

import net.librec.common.LibrecException;
import net.librec.conf.Configuration;
import net.librec.data.DataContext;
import net.librec.data.DataModel;
import net.librec.math.structure.SequentialAccessSparseMatrix;
import net.librec.math.structure.SparseTensor;
import net.librec.recommender.Recommender;
import net.librec.recommender.RecommenderContext;
import net.librec.recommender.TensorRecommender;


public class MultiTaskRecommender extends TensorRecommender{
    protected RecommenderContext sideContext;

    protected Configuration sideConf;
    protected SparseTensor sideTrainTensor;
    protected SparseTensor sideTestTensor;
    protected SparseTensor sideValidTensor;
    protected int sideNumDimensions;
    protected int[] sideDimensions;
    protected int sideNumUsers, sideNumItems;
    protected int sideNumFactors;
    protected int sideUserDimensions, sideItemDimensions;
    protected SequentialAccessSparseMatrix sideTrainMatrix;
    protected SequentialAccessSparseMatrix sideTestMatrix;

    public MultiTaskRecommender(Configuration conf, RecommenderContext context) throws LibrecException{
        sideConf = conf;
        this.sideContext = context;
        if (sideConf != sideContext.getConf()) {
            throw new LibrecException("conf of recommender context is not same init conf!");
        }
    }
    @Override
    protected void setup() throws LibrecException{
        super.setup();

        //setup side
        sideNumFactors = sideConf.getInt("rec.factor.number", 10);

        sideTrainTensor = (SparseTensor) getSideDataModel().getTrainDataSet();
        sideTestTensor = (SparseTensor) getSideDataModel().getTestDataSet();

        sideTrainMatrix = sideTrainTensor.rateMatrix();
        sideTestMatrix = sideTestTensor.rateMatrix();

        sideUserDimensions = sideTrainTensor.getUserDimension();
        sideItemDimensions = sideTrainTensor.getItemDimension();

        sideNumUsers = sideTrainTensor.dimensions()[sideUserDimensions];
        sideNumItems = sideTrainTensor.dimensions()[sideItemDimensions];

        int[] numDroppedSideItemsArray = new int[sideNumUsers];
        int maxSideNumTestItemsByUser = 0;
        for (int sideUserIdx = 0; sideUserIdx < sideNumUsers; ++ sideUserIdx) {
            numDroppedSideItemsArray[sideUserIdx] = sideNumItems - sideTrainMatrix.row(sideUserIdx).getNumEntries();
            int numTestSideItemsByUser = sideTestMatrix.row(sideUserIdx).getNumEntries();
            maxSideNumTestItemsByUser = maxSideNumTestItemsByUser < numTestSideItemsByUser ? numTestSideItemsByUser : maxSideNumTestItemsByUser;
        }

        int[] sideItemPurchasedCount = new int[sideNumItems];
        for (int sideItemIdx = 0; sideItemIdx < sideNumItems; ++sideItemIdx) {
            sideItemPurchasedCount[sideItemIdx] = sideTrainMatrix.column(sideItemIdx).getNumEntries()
                    + sideTestMatrix.column(sideItemIdx).getNumEntries();
        }

        sideConf.setInts("rec.eval.auc.dropped.num", numDroppedSideItemsArray);
        sideConf.setInt("rec.eval.key.test.max.num", maxSideNumTestItemsByUser);
        sideConf.setInt("rec.eval.item.num", sideTestMatrix.columnSize());
        sideConf.setInts("rec.eval.item.purchase.num", sideItemPurchasedCount);
    }
    protected void trainModel() throws LibrecException{}

    protected double predict(int[] keys) throws LibrecException{
        return predict(keys[0], keys[1]);
    }

    public DataModel getSideDataModel() {
        return sideContext.getDataModel();
    }
}
