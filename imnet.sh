export SPARK_WORKER_INSTANCES=4
export SPARKNET_HOME=/data02/nhe/SparkNet
export DEVICES=1
spark-submit --master yarn --deploy-mode cluster \
    --conf spark.yarn.appMasterEnv.SPARKNET_HOME=/data02/nhe/SparkNet \
    --conf spark.yarn.appMasterEnv.Redis=bdalab12 \
    --conf spark.yarn.appMasterEnv.GPU_HOSTS=bdalab12,bdalab13 \
    --conf spark.yarn.max.executor.failures=100 \
    --conf spark.driver.extraLibraryPath="${LD_LIBRARY_PATH}" \
    --conf spark.executorEnv.LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
    --driver-memory 5g \
    --executor-memory 100g \
    --executor-cores 5 \
    --driver-cores  4 \
    --num-executors ${SPARK_WORKER_INSTANCES} \
    --class apps.ImageNetApp   \
    sparknet-assembly-0.1-SNAPSHOT.jar  \
    4 hdfs://bdalab12:8020/imagenet  10 00 12000 128 false 125 false 40 2
