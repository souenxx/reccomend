package parameter;

import net.librec.common.LibrecException;
import net.librec.conf.Configuration;

import java.io.IOException;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;

public class gridSearch {
    protected Configuration conf;
    protected parameterAttribute parameterAttribute;
    protected ArrayList<Boolean> paraFlagList;
    protected ArrayList<Boolean> optFlagList;
    protected Deque<Integer> parameterIdxStack;
    protected double[] maxValue;
    protected double[] minValue;
    protected String[] parameterValues;


    public void setup() throws LibrecException, ClassNotFoundException{
        paraFlagList = new ArrayList<>();
        parameterIdxStack = new ArrayDeque<>();

        if (parameterAttribute == null) {
            Configuration.Resource resource = new Configuration.Resource("driver.parameter.properties");
            parameterAttribute = new parameterAttribute(conf, resource);
            parameterAttribute.setup("para.grid");
        }
        optFlagList = new ArrayList<>(parameterAttribute.getParaSize());
        parameterValues = new String[parameterAttribute.getParaSize()];
        for (int var = 0; var < parameterAttribute.getParaSize(); var++) { optFlagList.add(null); }

        for (int var = 0; var < parameterAttribute.getParaSize(); var++) {
            paraFlagList.add(false);
            if (!parameterAttribute.getParaOption(var).isEmpty()) {
                optFlagList.set(var, false);
            }
        }
        parameterIdxStack.push(0);
    }

    public boolean checkSearch() {
        if (parameterIdxStack.isEmpty()) return true;
        else return false;
    }

    public boolean schedule() throws ClassNotFoundException, LibrecException, IOException{
        Boolean finishSetFlag = false;
        while (!finishSetFlag && !parameterIdxStack.isEmpty()) {
            int stackSize = parameterIdxStack.size();
            int stackIdx = stackSize - 1;
            int paraIdx = parameterIdxStack.getFirst();
            int paraDataSize = parameterAttribute.getParaData(stackIdx).size();

            if (paraFlagList.get(stackIdx) == false) {
                paraFlagList.set(stackIdx, true);
                setParameter(stackIdx, paraIdx);

                if (stackSize != parameterAttribute.getParaSize()) parameterIdxStack.push(0);
                else finishSetFlag = true;

            } else {
                //check whether iterator
                String paraOption = parameterAttribute.getParaOption(stackIdx);
                if ((paraOption.isEmpty() && paraIdx < paraDataSize - 1)) {
                    paraIdx++;
                    parameterIdxStack.pop();
                    parameterIdxStack.push(paraIdx);
                    //reset flag of next prameter(same stackIdx) to false
                    paraFlagList.set(stackIdx, false);
                }
                else if (!paraOption.isEmpty() && !optFlagList.get(stackIdx)) {
                    paraIdx++;
                    parameterIdxStack.pop();
                    parameterIdxStack.push(paraIdx);
                    paraFlagList.set(stackIdx, false);
                } else {
                    paraFlagList.set(stackIdx, false);
                    if (!paraOption.isEmpty()) {
                        optFlagList.set(stackIdx, false);
                    }
                    parameterIdxStack.pop();
                }

            }
        }
        if (!parameterIdxStack.isEmpty()) {
            return true;
        }
        else return false;

    }

    protected void setParameter(int stackIdx, int paraIdx) throws ClassNotFoundException{
        String paraKey = parameterAttribute.getParaKey(stackIdx);
        Class<?> paraType = parameterAttribute.getParaType(stackIdx);
        String paraOption = parameterAttribute.getParaOption(stackIdx);

        if (!paraOption.isEmpty()) {
            try {
                String paraDataMinStr = parameterAttribute.getParaData(stackIdx).get(0);
                String paraDataMaxStr = parameterAttribute.getParaData(stackIdx).get(1);
                if (paraType == int.class) {
                    Integer setValue = Integer.valueOf(paraDataMinStr) + (Integer.valueOf(paraOption)* paraIdx);
                    parameterValues[stackIdx] = String.valueOf(setValue);
                    conf.setInt(paraKey, setValue);
                    if (setValue >= Integer.valueOf(paraDataMaxStr)) {optFlagList.set(stackIdx, true);}
                }
                if (paraType == double.class) {
                    Double setValue = Double.valueOf(paraDataMinStr)  + (Double.valueOf(paraOption) * (double)paraIdx);
                    parameterValues[stackIdx] = String.valueOf(setValue);
                    conf.setDouble(paraKey, setValue);
                    if (setValue >= Double.valueOf(paraDataMaxStr)) {optFlagList.set(stackIdx, true);}
                }
                if (paraType == float.class) {
                    Float setValue = Float.valueOf(paraDataMinStr) + (Float.valueOf(paraOption) * (float)paraIdx);
                    parameterValues[stackIdx] = String.valueOf(setValue);
                    conf.setFloat(paraKey, setValue);
                    if (setValue >= Float.valueOf(paraDataMaxStr)) {optFlagList.set(stackIdx, true);}
                }
                if (paraType == long.class) {
                    Long setValue = Long.valueOf(paraDataMinStr) + (Long.valueOf(paraOption) * (long)paraIdx);
                    parameterValues[stackIdx] = String.valueOf(setValue);
                    conf.setLong(paraKey, setValue);
                    if (setValue >= Long.valueOf(paraDataMaxStr)) {optFlagList.set(stackIdx, true);}
                }
            } catch (Exception e) {
            }
        } else {
            try {
                String paraDataStr = parameterAttribute.getParaData(stackIdx).get(paraIdx);
                parameterValues[stackIdx] = paraDataStr;
                if (paraType == int.class) {
                    conf.setInt(paraKey, Integer.valueOf(paraDataStr));
                }
                if (paraType == double.class) {
                    conf.setDouble(paraKey, Double.valueOf(paraDataStr));
                }
                if (paraType ==  float.class) {
                    conf.setFloat(paraKey, Float.valueOf(paraDataStr));
                }
                if (paraType == long.class) {
                    conf.setLong(paraKey, Long.valueOf(paraDataStr));
                }
                if (paraType == boolean.class) {
                    conf.setBoolean(paraKey, Boolean.valueOf(paraDataStr));
                }
            } catch (Exception e) {
            }
        }
    }

    public void setConf(Configuration conf) {this.conf = conf;}
    public String[] getParameterNames() {return parameterAttribute.getParaNameList();}
    public String[] getParameterValues() {return parameterValues;}

}
