# SparkNet
Distributed Neural Networks for Spark.
Details are available in the [paper](http://arxiv.org/abs/1511.06051).

## Quick Start
**Start a Spark cluster using our AMI**

1. Create an AWS secret key and access key. Instructions [here](http://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSGettingStartedGuide/AWSCredentials.html).
2. Run `export AWS_SECRET_ACCESS_KEY=` and `export AWS_ACCESS_KEY_ID=` with the relevant values.
3. Clone our repository locally.
4. Start a 5-worker Spark cluster on EC2 by running

        SparkNet/ec2/spark-ec2 --key-pair=key \
                               --identity-file=key.pem \
                               --region=eu-west-1 \
                               --zone=eu-west-1c \
                               --instance-type=g2.8xlarge \
                               --ami=ami-8b8c3ef8 \
                               --copy-aws-credentials \
                               --spark-version=1.5.0 \
                               --spot-price=1.5 \
                               --no-ganglia \
                               --user-data SparkNet/ec2/cloud-config.txt \
                               --slaves=5 \
                               launch sparknet
You will probably have to change several fields in this command.
For example, the flags `--key-pair` and `--identity-file` specify the key pair you will use to connect to the cluster.
The flag `--slaves` specifies the number of Spark workers.

**Train Cifar using SparkNet**

1. SSH to the Spark master as `root`.
2. Run `bash /root/SparkNet/data/cifar10/get_cifar10.sh` to get the Cifar data
3. Train Cifar on 5 workers using

        /root/spark/bin/spark-submit --class apps.CifarApp /root/SparkNet/target/scala-2.10/sparknet-assembly-0.1-SNAPSHOT.jar 5
4. That's all! Information is logged on the master in `/root/SparkNet/training_log*.txt`.

**Train ImageNet using SparkNet**

1. Obtain the ImageNet data by following the instructions [here](http://www.image-net.org/download-images) with

    ```
    wget http://.../ILSVRC2012_img_train.tar
    wget http://.../ILSVRC2012_img_val.tar
    ```
    This involves creating an account and submitting a request.
2. On the Spark master, create `~/.aws/credentials` with the following content:

    ```
    [default]
    aws_access_key_id=
    aws_secret_access_key=
    ```
    and fill in the two fields.
3. Copy this to the workers with `~/spark-ec2/copy-dir ~/.aws` (copy this command exactly because it is somewhat sensitive to the trailing backslashes and that kind of thing).
4. Create an Amazon S3 bucket with name `S3_BUCKET`.
5. Upload the ImageNet data in the appropriate format to S3 with the command

    ```
    python $SPARKNET_HOME/scripts/put_imagenet_on_s3.py $S3_BUCKET \
        --train_tar_file=/path/to/ILSVRC2012_img_train.tar \
        --val_tar_file=/path/to/ILSVRC2012_img_val.tar \
        --new_width=256 \
        --new_height=256
    ```
    This command resizes the images to 256x256, shuffles the training data, and tars the validation files into chunks.
6. Train ImageNet on 5 workers using

    ```
    /root/spark/bin/spark-submit --class apps.ImageNetApp /root/SparkNet/target/scala-2.10/sparknet-assembly-0.1-SNAPSHOT.jar 5 $S3_BUCKET
    ```

## Building your own AMI

1. Start an EC2 instance with Ubuntu 14.04 and a GPU instance type (e.g., g2.8xlarge). Suppose it has IP address xxx.xx.xx.xxx.
2. Connect to the node as `ubuntu`:

    ```
    ssh -i ~/.ssh/key.pem ubuntu@xxx.xx.xx.xxx
    ```
3. Install an editor `sudo apt-get install emacs`.
4. Open the file

    ```
    sudo emacs /root/.ssh/authorized_keys
    ```
    and delete everything before `ssh-rsa ...` so that you can connect to the node as `root`.
5. Close the connection with `exit`.
6. Connect to the node as `root`:

    ```
    ssh -i ~/.ssh/key.pem root@xxx.xx.xx.xxx
    ```
7. Intall CUDA-7.5. [The best instructions I've found are here](http://tleyden.github.io/blog/2015/11/22/cuda-7-dot-5-on-aws-gpu-instance-running-ubuntu-14-dot-04/).
8. `echo "/usr/local/cuda-7.5/targets/x86_64-linux/lib/" > /etc/ld.so.conf.d/cuda.conf`
9. `ldconfig`
10. Install sbt. [Instructions here](http://www.scala-sbt.org/0.13/docs/Installing-sbt-on-Linux.html).
11. `apt-get update`
12. `apt-get install awscli s3cmd`
13. Install Java `apt-get install openjdk-7-jdk`.
14. Clone the SparkNet repository `git clone https://github.com/amplab/SparkNet.git` in your home directory.
15. Build SparkNet with `cd ~/SparkNet` and `sbt assembly`.
16. Add the following to your `~/.bashrc`:

    ```
    export LD_LIBRARY_PATH=/usr/local/cuda-7.5/targets/x86_64-linux/lib
    export _JAVA_OPTIONS=-Xmx8g
    export SPARKNET_HOME=/root/SparkNet/
    ```
    Some of these paths may need to be adapted, but the `LD_LIBRARY_PATH` directory should contain `libcudart.so.7.5` (this file can be found with `locate libcudart.so.7.5` after running `updatedb`).
17. Create the file `~/.bash_profile` and add the following:

    ```
    if [ "$BASH" ]; then
      if [ -f ~/.bashrc ]; then
        . ~/.bashrc
      fi
    fi
    export JAVA_HOME=/usr/lib/jvm/java-7-openjdk-amd64
    ```
    Spark expects `JAVA_HOME` to be set in your `~/.bash_profile` and the launch script `SparkNet/ec2/spark-ec2` will give an error if it isn't there.
18. Clear your bash history `cat /dev/null > ~/.bash_history && history -c && exit`.
19. Now you can create an image of your instance, and you're all set! This is the procedure that we used to create our AMI.
