## MBML

```python
# train w/ MGAug
python train.py --net Conv4 --alg ProtoNet --dataset CUB --net_aug mask-layer-snip-mbml-small --min_width 0.97 --max_width 1. --shot_aug resize --num_shot 1 --num_way 5
# test
python test.py --net Conv4 --alg ProtoNet --dataset CUB --net_aug mask-layer-snip-mbml-small --min_width 0.97 --max_width 1. --shot_aug resize --num_shot 1 --num_way 5


# train w/o MGAug
python train.py --net Conv4 --alg ProtoNet --dataset CUB --num_shot 1 --num_way 5
# test
python test.py --net Conv4 --alg ProtoNet --dataset CUB --num_shot 1 --num_way 5
```

--net (Conv4, ResNet10)

--alg (ProtoNet)

--dataset (CUB)

--net_aug (mask-layer-snip-mbml, mask-layer-snip-mbml-small)

--shot_aug ( ,resize)

--num_shot (1, 5)

--num_way (1, 5)


## GBML

```python
# train w/ MGAug
python train.py --net Conv4 --alg FOMAML --dataset CUB --net_aug mask-layer-snip-fomaml-small --min_width 0.98 --max_width 1. --shot_aug resize --num_shot 1 --num_way 5
# test
python test.py --net Conv4 --alg FOMAML --dataset CUB --net_aug mask-layer-snip-fomaml-small --min_width 0.98 --max_width 1. --shot_aug resize --num_shot 1 --num_way 5


# train w/o MGAug
python train.py --net Conv4 --alg FOMAML --dataset CUB --num_shot 1 --num_way 5
# test
python test.py --net Conv4 --alg FOMAML --dataset CUB --num_shot 1 --num_way 5
```

--net (Conv4, ResNet10)

--alg (MAML, FOMAML)

--dataset (CUB)

--net_aug (mask-layer-snip-fomaml, mask-layer-snip-fomaml-small)

--shot_aug ( ,resize)

--num_shot (1, 5)

--num_way (1, 5)


## Dependencies

* Python >= 3.6
* Pytorch >= 1.2
* [Higher](https://github.com/facebookresearch/higher) 
* [Torchmeta](https://github.com/tristandeleu/pytorch-meta)
