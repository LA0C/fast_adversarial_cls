# fast_adversarial_cls
项目测试三种对抗训练模型（FGSM，PGD，FREE）在text classification上的表现，测试模型为textCNN。

## 环境
```
python3.6
torch 1.1+cu111
```

## 数据集
数据集来自THUCTC：http://thuctc.thunlp.org/<br>
文本类别涉及10个类别：categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']<br>
cnews.train.txt: 训练集(5000*10)<br>
cnews.val.txt: 验证集(500*10)<br>
cnews.test.txt: 测试集(1000*10)<br>

## 实现思路
在 TextCNN baseline 下，需要句子在 embedding 环节中生成干扰样本。本实验实现的思路是在 mini-batch 中梯度计算的过程利用上述方法生成对应的 attack干扰信息 。

三个超参数分别为epsilon、alpha、attack_iters

```
 epsilon = torch.tensor(0.1)
 alpha= 0.04
 attack_iters=5
```

## 运行步骤
```
python run.py train/test FGSM/PGD/FREE/baseline
```

模型和结果保存在saved_models/



## 训练结果

四种模型训练20轮后，在测试集上实验结果如下：

| **Model** | **Precision** | **Recall** | **F1-score** |
| --------- | ------------- | ---------- | ------------ |
| baseline  | 94.96         | 94.89      | 94.84        |
| FGSM      | 95.38         | 95.29      | 95.26        |
| PGD       | **95.38**     | **95.33**  | **95.30**    |
| FREE      | 95.33         | 95.22      | 95.18        |

四种模型训练消耗的时间(minutes)对比如下：

| **model** | **total_cost** | **mean_cost** |
| --------- | -------------- | ------------- |
| baseline  | 7.9            | 0.395         |
| FGSM      | 11.80          | 0.59          |
| PGD       | 27.13          | 1.3565        |
| FREE      | 23.64          | 1.182         |

## 结论
论文提出的FGSM方法，可以看出其分类效果跟其他两种方法基本相同，甚至比FREE表现更好一些。论文中主要强调FGSM的训练效率

# Reference

1.[FAST IS BETTER THAN FREE: REVISITING ADVERSARIAL TRAINING](https://arxiv.org/pdf/2001.03994.pdf)<br>
2.https://github.com/locuslab/fast_adversarial<br>
3.[Adversarial Training Methods for Semi-Supervised Text Classification.](https://arxiv.org/pdf/1605.07725.pdf)<br>
