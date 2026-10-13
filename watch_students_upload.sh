#!/bin/bash
# wait for node1 student (seed 7) to finish, fetch its SWA weight, create the Kaggle dataset with both students
cd "$(dirname "$0")"
until ssh -o ConnectTimeout=10 node1 'grep -q "^DONE" ~/projects/knee/train_coat_v6d_s7.log' 2>/dev/null; do sleep 120; done
scp -q node1:projects/knee/raptor_ft_v6d_s7_swa.pt kaggle_students_upload/ && \
kaggle datasets create -p kaggle_students_upload > kaggle_students_upload/create.out 2>&1
echo "finished rc=$?" >> kaggle_students_upload/create.out
