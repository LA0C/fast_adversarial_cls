# fast_adversarial_cls
项目测试三种对抗训练模型（FGSM，PGD，FREE）在text classification上的表现，测试模型为textCNN。主要参考论文Fast is better than free: Revisiting adversarial training涉及的三个对抗训练方法：<br>FGSM（Fast Gradient Sign Method）<br>PGD（projected gradient decent）<br>FREE（Free adversarial based on FGSM）

## 环境
python3.6
torch 1.1+cu111

## 数据集
文本类别涉及10个类别：categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']<br>
cnews.train.txt: 训练集(5000*10)<br>
cnews.val.txt: 验证集(500*10)<br>
cnews.test.txt: 测试集(1000*10)<br>

## 实现思路
CV任务的输入是连续的RGB的值，而NLP问题中，输入是离散的单词序列，一般以one-hot vector的形式呈现，如果直接在raw text上进行扰动，那么扰动的大小和方向可能都没什么意义。<br>
Goodfellow提出了可以在连续的embedding上做扰动，若在 TextCNN baseline 下，需要句子在 embedding 环节中生成干扰样本。本实验实现的思路是在 mini-batch 中梯度计算的过程利用上述方法生成对应的 attack干扰 信 息 ， 然后加入embedding中进行后续的卷积 + 池化学习

## 运行步骤
python run.py train/test FGSM/PGD/FREE/default <br>
模型和结果保存在saved_models/

## 结论
（1）对抗训练方法是有效的，对比TextCNN baseline，对抗训练方法平均有0.5%提升；<br>
（2）在三种对抗训练方法中，在本次实验中表现接近，效果最好是PGD方法，测试集准确率达到95.63%；<br>
（3）论文提出的FGSM方法，从表中可以看出其取得分 类效果跟其他两种方法是相同，甚至比FREE表现更好一些。论文中主要强调FGSM的训练效率
