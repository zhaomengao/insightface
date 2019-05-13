#!/usr/bin/env sh
#export MXNET_CUDNN_AUTOTUNE_DEFAULT=1

CWD=`pwd`
HDFS=hdfs://hobot-bigdata/
#set this to enable reading from hdfs
export CLASSPATH=$HADOOP_PREFIX/lib/classpath_hdfs.jar

export PYTHONPATH=${WORKING_PATH}:$PYTHONPATH
export PYTHONPATH=${WORKING_PATH}/python_package:$PYTHONPATH
cd ${WORKING_PATH}
hdfs dfs -get hdfs://hobot-bigdata-aliyun/user/mengjia.yan/platform/site-packages.tar.gz ${WORKING_PATH}
tar xf ${WORKING_PATH}/site-packages.tar.gz
hdfs dfs -get hdfs://hobot-bigdata-aliyun/user/xin.wang/mxnet_version/mxnet_quanti/ ${WORKING_PATH}
export PYTHONPATH=${WORKING_PATH}/mxnet_quanti/:$PYTHONPATH
export PYTHONPATH=${WORKING_PATH}/site-packages:$PYTHONPATH


export MXNET_CPU_WORKER_NTHREADS=24
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice


#hdfs dfs -get hdfs://hobot-bigdata/user/xuezhi.zhang/tmp/qian.zhang/mxnet.tar.gz ./
#tar -zxf mxnet.tar.gz
#rm mxnet.tar.gz

sh ${WORKING_PATH}/recognition/run_train_parall.sh
