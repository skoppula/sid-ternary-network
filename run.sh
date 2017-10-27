sentfilt=25
cache="/data/sls/scratch/skoppula/mfcc-nns/rsr-experiments/create_rsr_data_cache/trn_cache_sentfilt${sentfilt}/context_50frms_4mx/"
gpu=3

for t in 'no-twn' 'twn'; do
    for model in 'fcn' 'cnn' 'maxout' 'lcn' 'dsc2'; do
        echo "cd ~/get-shit-done; python rsr-run.py --gpu=$gpu --$t --model_name=$model --sentfilt=$sentfilt --cachedir=$cache"
    done
    echo
done


for t in 'twn'; do
    for model in 'fcn' 'cnn' 'maxout' 'lcn' 'dsc2'; do
        load_ckpt="train_log/sentfilt${sentfilt}_${model}_twnTrue/checkpoint"
        echo "python rsr-run.py --gpu=$gpu --$t --model_name=$model --sentfilt=$sentfilt --cachedir=$cache --load_ckpt=${load_ckpt}"
    done
done
