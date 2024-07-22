package com.graphhopper.http;

import com.graphhopper.GHRequest;
import com.graphhopper.GHResponse;
import com.graphhopper.PathWrapper;
import com.graphhopper.GraphHopper;
import com.graphhopper.routing.util.EncodingManager;
import com.graphhopper.util.GPXEntry;
import com.graphhopper.util.InstructionList;
import com.graphhopper.util.PointList;
import com.graphhopper.reader.osm.GraphHopperOSM;


import java.io.*;
import java.util.*;

import static sun.swing.MenuItemLayoutHelper.max;

public class GraphHopperMainGen {
    static Map<String, Integer> keyChanceMap = new HashMap<String, Integer>();

    public static void main(String[] args) {
        // create one GraphHopper instance
        GraphHopper hopper = new GraphHopperOSM().forServer();
        hopper.setDataReaderFile(".\\map-data\\beijing.osm");
        // where to store graphhopper files?
        hopper.setGraphHopperLocation(".\\graph-cache");
        hopper.setEncodingManager(new EncodingManager("car"));

        // now this can take minutes if it imports or a few seconds for loading
        // of course this is dependent on the area you import
        hopper.importOrLoad();

        // parameter setting 1: sparsity=1000, thre=10000
        // parameter setting 2: sparsity=2000, thre=20000
        boolean od_prop = false;
        int trace_num=1000000;  // 生成轨迹的条数
        double GAP = 2.5; // 调整时间间隔，不用变
        int trace_idx = 0;
        int sparsity = 2000;  // 平均1km割一段
        int thre = 10000;  //10km偏移1km
        long ts_ini = 1538841600;  // 2018-10-07
        String PATH_OUT = "D:\\DeepMapMatching\\data\\tencent\\preprocessed\\step1_gen_para2\\generate_" + trace_num + '_' + GAP + ".csv";
        String PATH_ORI_OUT = "D:\\DeepMapMatching\\data\\tencent\\preprocessed\\step1_gen_para2\\generate_" + trace_num + '_' + GAP + ".csv";
//        String[] mode = new String[]{"fastest", "shortest", "short_fastest"};
        List<String> mode = Arrays.asList("fastest");

        //double lon_min = 116.3612830071808;
        //double lon_max = 116.46;
        //double lat_min = 39.8921412898523;
        ///double lat_max = 39.95556678758318;

        double lon_min = 116.269
        double lon_max = 116.498
        double lat_min = 39.827
        double lat_max = 39.885

        Random r = new Random();

        if (od_prop) {
            try {
                BufferedReader reader = new BufferedReader(new FileReader("D:\\DeepMapMatching\\data\\tencent\\preprocessed\\step4_addnoise\\beijing-part_10min_2min_2km_1km\\timegap-60_noise-gaussian_sigma-100_dup-8\\od_distribute.csv"));
                String line = null;
                while((line=reader.readLine())!=null){
                    String item[] = line.split(",");//CSV格式文件为逗号分隔符文件，这里根据逗号切分

                    String last = item[item.length-1];
                    int value = Integer.parseInt(last);//如果是数值，可以转化为数值
                    keyChanceMap.put(join(",", Arrays.copyOfRange(item, 0, 8)), value);
//                    System.out.println(last);
                }
            } catch (Exception e) {
                e.printStackTrace();
                return;
            }
        }

        while(trace_idx < trace_num) {
            if (trace_idx % 1000 == 0){
                System.out.printf("generate number: %d \n", trace_idx);
            }
//            System.out.println("***************************************");
            //生成随机起点终点
            double f_lat;
            double f_lon;
            double t_lat;
            double t_lon;
            if (od_prop){
                String item[] = chanceSelect(keyChanceMap).split(",");
                lat_min = Double.parseDouble(item[0]);
                lon_min = Double.parseDouble(item[1]);
                lat_max = Double.parseDouble(item[2]);
                lon_max = Double.parseDouble(item[3]);
                f_lat = lat_min + r.nextDouble() * (lat_max - lat_min);
                f_lon = lon_min + r.nextDouble() * (lon_max - lon_min);

                lat_min = Double.parseDouble(item[4]);
                lon_min = Double.parseDouble(item[5]);
                lat_max = Double.parseDouble(item[6]);
                lon_max = Double.parseDouble(item[7]);
                t_lat = lat_min + r.nextDouble() * (lat_max - lat_min);
                t_lon = lon_min + r.nextDouble() * (lon_max - lon_min);
            }
            else{
                 f_lat = lat_min + r.nextDouble() * (lat_max - lat_min);
                 f_lon = lon_min + r.nextDouble() * (lon_max - lon_min);
                 t_lat = lat_min + r.nextDouble() * (lat_max - lat_min);
                 t_lon = lon_min + r.nextDouble() * (lon_max - lon_min);
            }


            String weight = mode.get((int) (Math.random()* mode.size()));
            // simple configuration of the request object, see the GraphHopperServlet classs for more possibilities.
            GHRequest req = new GHRequest(f_lat,f_lon, t_lat, t_lon).
                    setWeighting(weight).  // 这里随机用一种模式，不要都固定为fastest，参考https://github.com/graphhopper/graphhopper/blob/master/docs/web/api-doc.md
                    setVehicle("car").
                    setLocale(Locale.US);
            GHResponse rsp = hopper.route(req);

            // first check for errors
            if (rsp.hasErrors()) {
                // handle them!
                System.out.println(rsp.getErrors());
                System.out.println("ori trace error stop");
            }
            else {
                // use the best path, see the GHResponse class for more possibilities.
                PathWrapper path = rsp.getBest();
                InstructionList instructionList = path.getInstructions();
                List<GPXEntry> gpxList = instructionList.createGPXList();

                // points, distance in meters and time in millis of the full path
                PointList pointList = path.getPoints();
                int trace_len = pointList.getSize();
                double trace_dist = path.getDistance();
//                System.out.println(trace_dist);
                long timeInMs = path.getTime();//' ' + timeInMs +
                timeInMs = (int) (timeInMs * GAP);
                long ts_start = ts_ini + r.nextInt(24 * 3600 - (int) timeInMs / 1000 - 1);

                // 切分
                int split_point_num = (int)(Math.random()*trace_dist/sparsity)+1;
                int[] split_point = getRandomArrayByIndex(split_point_num, pointList.getSize());  // 第2个点到倒数第2个点之间选取一些分割点
                Arrays.sort(split_point);
                int[] tempArray=new int[split_point.length+1];
                for(int i=0;i<split_point.length;i++)
                {
                    tempArray[i]=split_point[i];
                }
                tempArray[split_point.length]=pointList.size()-1;
                split_point=tempArray;

                double start_lat = pointList.getLat(0);
                double start_lon = pointList.getLon(0);
                ArrayList<ArrayList<String>> subtraces = new ArrayList<ArrayList<String>>();
                for (int i = 0; i <= split_point_num; i++){
                    double lat_noise = (r.nextDouble() - 1)*2 * trace_dist / thre * 0.01;
                    double lon_noise = (r.nextDouble() - 1)*2 * trace_dist / thre * 0.01;
//                    System.out.println(lat_noise);
                    double end_lat = pointList.getLat(split_point[i]);
                    double end_lon = pointList.getLon(split_point[i]);
                    if (i < split_point_num) {
                        end_lat = end_lat + lat_noise;
                        end_lon = end_lon + lon_noise;
                    }
                    TwoTuple<ArrayList<String>, Long> subtrace = genTrace(mode, start_lat, start_lon, end_lat, end_lon, hopper, GAP, r, trace_idx, ts_start);
                    subtraces.add(subtrace.getFirst());
                    start_lat = end_lat;
                    start_lon = end_lon;
                    ts_start = ts_start + subtrace.getSecond();
                }

//                System.out.println("split_point_num: " + Integer.toString(split_point_num));

                boolean flag = true;
                for (int i=0; i<subtraces.size(); i++){
                    if (subtraces.get(i).size() == 0){
                        flag = false;
                    }
                }
                if (flag) {
//                    // 保存原始轨迹
//                    try {
//                        try (Writer writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(PATH_ORI_OUT, true), "utf-8"))) {
//                            for (int i = 0; i < pointList.size(); i++) {
//                                writer.write("generate_" + Integer.toString(trace_idx) + ",0," + Long.toString(ts_start + (int) (gpxList.get(i).getTime() / 1000 * GAP)) + ',' + Double.toString(pointList.getLat(i)) + ',' + Double.toString(pointList.getLon(i)) + '\n');
//                            }
//                        }
//                    } catch (IOException ex) {
//                        System.out.println("error save file");
//                        throw new RuntimeException(ex);
//                    }

                    // 保存噪声轨迹
                    try {
                        try (Writer writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(PATH_OUT, true), "utf-8"))) {
                            for (int i = 0; i < subtraces.size(); i++) {
                                ArrayList<String> tmptrace = subtraces.get(i);
                                for (int j = 0; j < tmptrace.size(); j++) {
                                    writer.write(tmptrace.get(j));
                                }
                            }
                        }
                    } catch (IOException ex) {
                        System.out.println("error save file");
                        throw new RuntimeException(ex);
                    }

                    trace_idx++;
                }
            }

        }

    }

    public static String join(String join,String[] strAry){
        StringBuffer sb=new StringBuffer();
        for(int i=0,len =strAry.length;i<len;i++){
            if(i==(len-1)){
                sb.append(strAry[i]);
            }else{
                sb.append(strAry[i]).append(join);
            }
        }
        return sb.toString();
    }

    public static String chanceSelect(Map<String, Integer> keyChanceMap) {
        if (keyChanceMap == null || keyChanceMap.size() == 0)
            return null;

        Integer sum = 0;
        for (Integer value : keyChanceMap.values()) {
            sum += value;
        }
        // 从1开始
        Integer rand = new Random().nextInt(sum) + 1;

        for (Map.Entry<String, Integer> entry : keyChanceMap.entrySet()) {
            rand -= entry.getValue();
            // 选中
            if (rand <= 0) {
                String item = entry.getKey();
                return item;
            }
        }
        return null;
    }

    public static class TwoTuple<A, B> {

        public final A first;
        public final B second;

        public TwoTuple(A a, B b){
            first = a;
            second = b;
        }

        public A getFirst(){
            return first;
        }
        public B getSecond(){
            return second;
        }

        public String toString(){
            return "(" + first + ", " + second + ")";
        }

    }

    public static TwoTuple<ArrayList<String>, Long> genTrace(List<String> mode, double f_lat, double f_lon, double t_lat, double t_lon, GraphHopper hopper, double GAP, Random r, int trace_idx, long ts_start){
        String weight = mode.get((int) (Math.random()* mode.size()));
        // simple configuration of the request object, see the GraphHopperServlet classs for more possibilities.
        GHRequest req = new GHRequest(f_lat,f_lon, t_lat, t_lon).
                setWeighting(weight).  // 这里随机用一种模式，不要都固定为fastest，参考https://github.com/graphhopper/graphhopper/blob/master/docs/web/api-doc.md
                setVehicle("car").
                setLocale(Locale.US);
        GHResponse rsp = hopper.route(req);

        ArrayList<String> trace = new ArrayList<String>();
        long timeInS = 0;
        // first check for errors
        if (rsp.hasErrors()) {
            // handle them!
            // rsp.getErrors()
            System.out.println("error stop");
        }
        else {
            // use the best path, see the GHResponse class for more possibilities.
            PathWrapper path = rsp.getBest();
            InstructionList instructionList = path.getInstructions();
            List<GPXEntry> gpxList = instructionList.createGPXList();

            // points, distance in meters and time in millis of the full path
            PointList pointList = path.getPoints();
            long timeInMs = path.getTime();//' ' + timeInMs +
            timeInS = (long) (timeInMs * GAP / 1000);

            for (int i = 0; i < pointList.size(); i++) {
                trace.add("generate_" + Integer.toString(trace_idx) + ",0," + Long.toString(ts_start + (int) (gpxList.get(i).getTime()/1000*GAP)) + ',' + Double.toString(pointList.getLat(i)) + ',' + Double.toString(pointList.getLon(i)) + '\n');
            }
        }
        TwoTuple<ArrayList<String>, Long> result = new TwoTuple<ArrayList<String>, Long>(trace, timeInS);
        return result;
    }

    public static int[] getRandomArrayByIndex(int num,int scope){
        //1.获取scope范围内的所有数值，并存到数组中
        int[] randomArray=new int[scope];
        for(int i=1;i<randomArray.length-1;i++){
            randomArray[i]=i;
        }

        //2.从数组random中取数据，取过后的数改为-1
        int[] numArray=new int[num];//存储num个随机数
        int i=0;
        while(i<numArray.length){
            int index=(int)(Math.random()*scope);
            if(randomArray[index]!=-1){
                numArray[i]=randomArray[index];
                randomArray[index]=-1;
                i++;
            }
        }

        return numArray;
    }

}
