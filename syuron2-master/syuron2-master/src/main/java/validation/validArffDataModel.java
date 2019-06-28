package validation;

import net.librec.common.LibrecException;
import net.librec.conf.Configuration;
import net.librec.data.DataContext;
import net.librec.data.DataModel;
import net.librec.data.DataSplitter;
import net.librec.data.model.ArffDataModel;
import net.librec.math.structure.DataSet;
import net.librec.math.structure.MatrixEntry;
import net.librec.math.structure.SequentialAccessSparseMatrix;
import net.librec.math.structure.SparseTensor;

import javax.xml.crypto.Data;
import java.util.List;


public class validArffDataModel extends ArffDataModel implements DataModel{

    protected DataContext validContext;
    protected DataSet validTrainDataSet;
    protected DataSet validTestDataset;
    protected validKCVDataSplitter validDataSplitter;
    protected DataSet totalTrainDataSet;
    protected DataSet totalTestDataSet;
    public validArffDataModel() {}

    public validArffDataModel(Configuration conf) {
        this.conf = conf;
    }

    /**
     * Build validArffModel common in ArffDataModel
     * @throws LibrecException
     */
    @Override
    public void buildConvert() throws LibrecException {
        super.buildConvert();
    }

    /**
     * Build Splitter
     * @throws LibrecException
     */
    @Override
    protected void buildSplitter() throws LibrecException {
        super.buildSplitter();
    }

    protected void amendSplitter() {
        if (dataConvertor != null && dataSplitter != null) {
            SparseTensor totalTensor = dataConvertor.getSparseTensor();
            SequentialAccessSparseMatrix totalTestMatrix = ((validKCVDataSplitter)dataSplitter).getTotalTestMatrix();
            SparseTensor totalTrainTensor = totalTensor.clone();

            int[] dimensions = totalTrainTensor.dimensions();
            SparseTensor totalTestTensor = new SparseTensor(dimensions);
            totalTestTensor.setUserDimension(totalTrainTensor.getUserDimension());
            totalTestTensor.setItemDimension(totalTrainTensor.getItemDimension());

            for (MatrixEntry me : totalTestMatrix) {
                int u = me.row();
                int i = me.column();

                List<Integer> indices = totalTensor.getIndices(u, i);

                for (int index : indices) {
                    int[] keys = totalTensor.keys(index);
                    try {
                        totalTestTensor.set(totalTensor.value(index), keys);
                        totalTrainTensor.remove(keys);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            }
            totalTrainDataSet = totalTrainTensor;
            totalTestDataSet = totalTestTensor;
            if (conf.get("data.split.valid").equals("valid")) {
                validAmendSplitter();
            } else if (conf.get("data.split.valid").equals("test")) {
                trainDataSet = totalTrainTensor;
                testDataSet = totalTestTensor;
            }
        }
    }

    protected void validAmendSplitter() {
        SparseTensor totalValidTensor = (SparseTensor) totalTrainDataSet;
        SequentialAccessSparseMatrix testMatrix = dataSplitter.getTestData();
        SparseTensor trainTensor = totalValidTensor.clone();

        int[] dimensions = dataConvertor.getSparseTensor().dimensions();
        SparseTensor testTensor = new SparseTensor(dimensions);
        testTensor.setUserDimension(dataConvertor.getSparseTensor().getUserDimension());
        testTensor.setItemDimension(dataConvertor.getSparseTensor().getItemDimension());

        for (MatrixEntry me : testMatrix) {
            int u = me.row();
            int i = me.column();

            List<Integer> indices = totalValidTensor.getIndices(u, i);

            for (int index : indices) {
                int[] keys = totalValidTensor.keys(index);
                try {
                    testTensor.set(totalValidTensor.value(index), keys);
                    trainTensor.remove(keys);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
        trainDataSet = trainTensor;
        testDataSet = testTensor;
        LOG.info("Data cardinality of validation training is " + trainDataSet.size());
        LOG.info("Data cardinality of validation testing is " + testDataSet.size());
    }
    @Override
    public void nextFold(){
        amendSplitter();
    }

    public void setDatasplitter(DataSplitter dataSplitter) {this.dataSplitter = dataSplitter; }


}
