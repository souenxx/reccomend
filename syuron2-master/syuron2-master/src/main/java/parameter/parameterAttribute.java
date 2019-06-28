package parameter;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import com.google.common.collect.HashBiMap;

import net.librec.common.LibrecException;
import net.librec.conf.Configuration;

public class parameterAttribute {
    protected  String[] paraNameList;
    protected  String[] paraKeyList;
    protected  ArrayList<ArrayList<String>> paraDataList;
    protected  ArrayList<String> paraTypeStringList;
    protected  HashBiMap<String, Integer> paraIndexPairs;
    /**
     * parameter option is iterator for grid search
     */
    private String[] paraOptionList;
    private int paraSize;

    private static final Map<String, Class<?>> PRIMITIVE_TYPE_MAP = new HashMap<String, Class<?>>();
    protected Configuration conf;

    //setup driver parameter properties
    parameterAttribute(Configuration conf, Configuration.Resource resource) {
        this.conf = conf;
        this.conf.addResource(resource);
    }
    protected void setup(String keyClassName) throws LibrecException, ClassNotFoundException {
        paraNameList = conf.get("parameter.nameList").split(",");
        int paraNLSize = paraNameList.length;
        paraSize = paraNLSize;
        paraDataList = new ArrayList<>(paraNLSize);
        paraTypeStringList = new ArrayList<>(paraNLSize);
        for (int var0 = 0; var0 < paraSize; var0++) {
            paraDataList.add(null);
            paraTypeStringList.add(null);
        }
        paraKeyList = new String[paraNLSize];
        paraOptionList = new String[paraNLSize];

        //create paraIndexPairs
        paraIndexPairs = HashBiMap.create();
        for (int paraIdx = 0; paraIdx < paraNameList.length; paraIdx++) {
           paraIndexPairs.put(paraNameList[paraIdx], paraIdx);
        }

        for (Map.Entry<String, Integer> paraIdxPairs : paraIndexPairs.entrySet()) {
            String paraName = paraIdxPairs.getKey();
            String paraSetName = keyClassName + "." + paraName + ".";
            int paraIdx = paraIdxPairs.getValue();

            //set paraKey
            paraKeyList[paraIdx] = conf.get(paraName);

            //set paraData
            String[] paraData = conf.get(paraSetName + "data").split(",");
            paraDataList.set(paraIdx, new ArrayList<String>(Arrays.asList(paraData)));

            //set paraType
            String paraTypeStr = conf.get(paraSetName + "type");
            String paraTypeString = getTypeString(paraTypeStr);
            paraTypeStringList.set(paraIdx, paraTypeString);

            //set paraOption
            paraOptionList[paraIdx] = conf.get(paraSetName + "option");
        }
    }

    static {
        registerPrimitiveType("byte", byte.class);
        registerPrimitiveType("short", short.class);
        registerPrimitiveType("int", int.class);
        registerPrimitiveType("long", long.class);
        registerPrimitiveType("boolean", boolean.class);
        registerPrimitiveType("float", float.class);
        registerPrimitiveType("double", double.class);
        registerPrimitiveType("char", char.class);
    }

    private static void registerPrimitiveType(String typeName, Class<?> clazz) {
        PRIMITIVE_TYPE_MAP.put(typeName, clazz);
    }

    private Class<?> getType(String className) throws ClassNotFoundException {
       if (PRIMITIVE_TYPE_MAP.containsKey(className)) {
           return (Class<?>) PRIMITIVE_TYPE_MAP.get(className);
       }
       return (Class<?>) Class.forName(className);
    }

    private String getTypeString(String className) throws ClassNotFoundException {
        if (PRIMITIVE_TYPE_MAP.containsKey(className))
            return className;
        else
            return null;
    }

    public int getParaSize() {return paraSize;}

    public ArrayList<String> getParaData(int idx) {return paraDataList.get(idx);}

    public Class<?> getParaType(int idx) throws ClassNotFoundException {return getType(paraTypeStringList.get(idx));}

    public String getParaKey(int idx) {return paraKeyList[idx];}

    public String getParaOption(int idx) {return paraOptionList[idx];}

    public String[] getParaNameList() {return paraNameList;}

}
