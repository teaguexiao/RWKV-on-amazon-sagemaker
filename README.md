# RWKV-on-amazon-sagemaker
This is a sample about how to run RWKV on Amazon SageMaker, include finetune and inference, this sample is only for demo use.

We use [s5cmd](https://github.com/peak/s5cmd), a fast tool for S3 download and upload in parallel, to download pretrained model files from S3 and upload fine-tuned model to S3.

This sample code is tested on [Amazon SageMaker](https://aws.amazon.com/pm/sagemaker/) by creating a training job with p4d.24xlarge (8 * A100, 40GB), you could run this with p4de.24xlarge (8 * A100, 80GB) of course.

This repo including RWKV fine-tuning on SageMaker training job and inferening on SageMaker endpoints.