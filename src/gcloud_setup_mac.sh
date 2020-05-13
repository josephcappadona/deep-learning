curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init


export IMAGE_FAMILY="tf2-latest-gpu" # or "tf2-latest-cpu" for non-GPU instances
export ZONE="us-east4-b"
export INSTANCE_NAME="my-instance"
export INSTANCE_TYPE="n1-highmem-8" # budget: "n1-highmem-4"

# budget: 'type=nvidia-tesla-k80,count=1'
gcloud compute instances create $INSTANCE_NAME \
        --zone=$ZONE \
        --image-family=$IMAGE_FAMILY \
        --image-project=deeplearning-platform-release \
        --maintenance-policy=TERMINATE \
        --accelerator="type=nvidia-tesla-p4,count=1" \
        --machine-type=$INSTANCE_TYPE \
        --boot-disk-size=200GB \
        --metadata="install-nvidia-driver=True" \
        --preemptible
